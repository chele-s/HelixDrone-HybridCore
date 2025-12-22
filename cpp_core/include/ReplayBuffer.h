#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <atomic>
#include <memory>
#include <immintrin.h>
#include <cstring>

#ifdef _MSC_VER
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define ALIGNED_ALLOC(alignment, size) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#endif

namespace helix {

template<typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    
    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
    
    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        size_t bytes = n * sizeof(T);
        size_t aligned_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);
        void* ptr = ALIGNED_ALLOC(Alignment, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) noexcept {
        ALIGNED_FREE(p);
    }
    
    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

class SumTree {
public:
    explicit SumTree(size_t capacity) 
        : capacity_(capacity), 
          treeSize_(2 * capacity - 1),
          tree_(treeSize_, 0.0),
          dataPointer_(0) {
        leafStart_ = capacity_ - 1;
    }
    
    void update(size_t treeIdx, double priority) noexcept {
        double change = priority - tree_[treeIdx];
        tree_[treeIdx] = priority;
        propagateUp(treeIdx, change);
    }
    
    void updateBatch(const int32_t* treeIndices, const double* priorities, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            update(static_cast<size_t>(treeIndices[i]), priorities[i]);
        }
    }
    
    size_t add(double priority) noexcept {
        size_t treeIdx = dataPointer_ + leafStart_;
        update(treeIdx, priority);
        size_t dataIdx = dataPointer_;
        dataPointer_ = (dataPointer_ + 1) % capacity_;
        return dataIdx;
    }
    
    void addBatch(const double* priorities, size_t* dataIndices, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            dataIndices[i] = add(priorities[i]);
        }
    }
    
    void get(double value, size_t& treeIdx, double& priority, size_t& dataIdx) const noexcept {
        size_t parentIdx = 0;
        while (true) {
            size_t leftIdx = (parentIdx << 1) + 1;
            if (leftIdx >= treeSize_) break;
            size_t rightIdx = leftIdx + 1;
            if (value <= tree_[leftIdx]) {
                parentIdx = leftIdx;
            } else {
                value -= tree_[leftIdx];
                parentIdx = rightIdx;
            }
        }
        treeIdx = parentIdx;
        priority = tree_[parentIdx];
        dataIdx = parentIdx - leafStart_;
    }
    
    void sampleStratified(
        size_t batchSize,
        size_t* treeIndices,
        double* priorities,
        size_t* dataIndices,
        std::mt19937& rng
    ) const noexcept {
        double totalP = tree_[0];
        if (totalP < 1e-8) totalP = 1e-8;
        double segment = totalP / static_cast<double>(batchSize);
        
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < batchSize; ++i) {
            double low = segment * static_cast<double>(i);
            double value = low + dist(rng) * segment;
            value = std::min(value, totalP - 1e-8);
            
            size_t treeIdx, dataIdx;
            double priority;
            get(value, treeIdx, priority, dataIdx);
            
            treeIndices[i] = treeIdx;
            priorities[i] = std::max(priority, 1e-8);
            dataIndices[i] = dataIdx % capacity_;
        }
    }
    
    double totalPriority() const noexcept { return tree_[0]; }
    
    double maxPriority() const noexcept {
        double maxP = 0.0;
        const double* leafs = tree_.data() + leafStart_;
        size_t n = capacity_;
        
        size_t i = 0;
        #ifdef __AVX__
        __m256d maxVec = _mm256_setzero_pd();
        for (; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(leafs + i);
            maxVec = _mm256_max_pd(maxVec, v);
        }
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, maxVec);
        maxP = std::max({tmp[0], tmp[1], tmp[2], tmp[3]});
        #endif
        
        for (; i < n; ++i) {
            maxP = std::max(maxP, leafs[i]);
        }
        return maxP;
    }
    
    double minPriority() const noexcept {
        double minP = std::numeric_limits<double>::max();
        const double* leafs = tree_.data() + leafStart_;
        for (size_t i = 0; i < capacity_; ++i) {
            if (leafs[i] > 0.0) minP = std::min(minP, leafs[i]);
        }
        return minP == std::numeric_limits<double>::max() ? 1.0 : minP;
    }
    
    size_t capacity() const noexcept { return capacity_; }

private:
    void propagateUp(size_t treeIdx, double change) noexcept {
        while (treeIdx != 0) {
            treeIdx = (treeIdx - 1) >> 1;
            tree_[treeIdx] += change;
        }
    }
    
    size_t capacity_;
    size_t treeSize_;
    size_t leafStart_;
    AlignedVector<double> tree_;
    size_t dataPointer_;
};

struct alignas(64) TransitionBlock {
    float* states;
    float* actions;
    float* rewards;
    float* nextStates;
    float* dones;
    size_t stateDim;
    size_t actionDim;
    size_t capacity;
    
    TransitionBlock(size_t cap, size_t sDim, size_t aDim) 
        : stateDim(sDim), actionDim(aDim), capacity(cap) {
        size_t stateBytes = cap * sDim * sizeof(float);
        size_t actionBytes = cap * aDim * sizeof(float);
        size_t scalarBytes = cap * sizeof(float);
        
        stateBytes = (stateBytes + 63) & ~63;
        actionBytes = (actionBytes + 63) & ~63;
        scalarBytes = (scalarBytes + 63) & ~63;
        
        states = static_cast<float*>(ALIGNED_ALLOC(64, stateBytes));
        actions = static_cast<float*>(ALIGNED_ALLOC(64, actionBytes));
        rewards = static_cast<float*>(ALIGNED_ALLOC(64, scalarBytes));
        nextStates = static_cast<float*>(ALIGNED_ALLOC(64, stateBytes));
        dones = static_cast<float*>(ALIGNED_ALLOC(64, scalarBytes));
        
        std::memset(states, 0, stateBytes);
        std::memset(actions, 0, actionBytes);
        std::memset(rewards, 0, scalarBytes);
        std::memset(nextStates, 0, stateBytes);
        std::memset(dones, 0, scalarBytes);
    }
    
    ~TransitionBlock() {
        ALIGNED_FREE(states);
        ALIGNED_FREE(actions);
        ALIGNED_FREE(rewards);
        ALIGNED_FREE(nextStates);
        ALIGNED_FREE(dones);
    }
    
    TransitionBlock(const TransitionBlock&) = delete;
    TransitionBlock& operator=(const TransitionBlock&) = delete;
    
    void store(size_t idx, const float* s, const float* a, float r, const float* ns, float d) noexcept {
        std::memcpy(states + idx * stateDim, s, stateDim * sizeof(float));
        std::memcpy(actions + idx * actionDim, a, actionDim * sizeof(float));
        rewards[idx] = r;
        std::memcpy(nextStates + idx * stateDim, ns, stateDim * sizeof(float));
        dones[idx] = d;
    }
    
    void storeBatch(size_t startIdx, const float* s, const float* a, const float* r, 
                    const float* ns, const float* d, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            size_t idx = (startIdx + i) % capacity;
            std::memcpy(states + idx * stateDim, s + i * stateDim, stateDim * sizeof(float));
            std::memcpy(actions + idx * actionDim, a + i * actionDim, actionDim * sizeof(float));
            rewards[idx] = r[i];
            std::memcpy(nextStates + idx * stateDim, ns + i * stateDim, stateDim * sizeof(float));
            dones[idx] = d[i];
        }
    }
    
    void gather(const size_t* indices, size_t count,
                float* outStates, float* outActions, float* outRewards,
                float* outNextStates, float* outDones) const noexcept {
        for (size_t i = 0; i < count; ++i) {
            size_t idx = indices[i];
            std::memcpy(outStates + i * stateDim, states + idx * stateDim, stateDim * sizeof(float));
            std::memcpy(outActions + i * actionDim, actions + idx * actionDim, actionDim * sizeof(float));
            outRewards[i] = rewards[idx];
            std::memcpy(outNextStates + i * stateDim, nextStates + idx * stateDim, stateDim * sizeof(float));
            outDones[i] = dones[idx];
        }
    }
};

class PrioritizedReplayBuffer {
public:
    PrioritizedReplayBuffer(
        size_t capacity,
        size_t stateDim,
        size_t actionDim,
        double alpha = 0.6,
        double betaStart = 0.4,
        size_t betaFrames = 100000,
        double epsilon = 1e-6
    ) : capacity_(capacity),
        stateDim_(stateDim),
        actionDim_(actionDim),
        alpha_(alpha),
        betaStart_(betaStart),
        betaFrames_(betaFrames),
        epsilon_(epsilon),
        frame_(1),
        size_(0),
        maxPriority_(1.0),
        tree_(capacity),
        data_(capacity, stateDim, actionDim),
        rng_(std::random_device{}()),
        tempTreeIndices_(capacity),
        tempPriorities_(capacity),
        tempDataIndices_(capacity) {}
    
    double beta() const noexcept {
        double progress = static_cast<double>(frame_) / static_cast<double>(betaFrames_);
        return std::min(1.0, betaStart_ + (1.0 - betaStart_) * progress);
    }
    
    void push(const float* state, const float* action, float reward, 
              const float* nextState, float done) noexcept {
        double priority = std::pow(maxPriority_, alpha_);
        size_t dataIdx = tree_.add(priority);
        data_.store(dataIdx, state, action, reward, nextState, done);
        size_ = std::min(size_ + 1, capacity_);
    }
    
    void pushBatch(const float* states, const float* actions, const float* rewards,
                   const float* nextStates, const float* dones, size_t batchSize) noexcept {
        double priority = std::pow(maxPriority_, alpha_);
        
        for (size_t i = 0; i < batchSize; ++i) {
            size_t dataIdx = tree_.add(priority);
            data_.store(dataIdx, 
                       states + i * stateDim_,
                       actions + i * actionDim_,
                       rewards[i],
                       nextStates + i * stateDim_,
                       dones[i]);
        }
        size_ = std::min(size_ + batchSize, capacity_);
    }
    
    struct SampleResult {
        AlignedVector<float> states;
        AlignedVector<float> actions;
        AlignedVector<float> rewards;
        AlignedVector<float> nextStates;
        AlignedVector<float> dones;
        AlignedVector<float> weights;
        AlignedVector<int32_t> treeIndices;
        size_t batchSize;
        size_t stateDim;
        size_t actionDim;
    };
    
    SampleResult sample(size_t batchSize) {
        SampleResult result;
        result.batchSize = batchSize;
        result.stateDim = stateDim_;
        result.actionDim = actionDim_;
        
        result.states.resize(batchSize * stateDim_);
        result.actions.resize(batchSize * actionDim_);
        result.rewards.resize(batchSize);
        result.nextStates.resize(batchSize * stateDim_);
        result.dones.resize(batchSize);
        result.weights.resize(batchSize);
        result.treeIndices.resize(batchSize);
        
        tree_.sampleStratified(
            batchSize,
            tempTreeIndices_.data(),
            tempPriorities_.data(),
            tempDataIndices_.data(),
            rng_
        );
        
        data_.gather(
            tempDataIndices_.data(), batchSize,
            result.states.data(),
            result.actions.data(),
            result.rewards.data(),
            result.nextStates.data(),
            result.dones.data()
        );
        
        double totalP = std::max(tree_.totalPriority(), 1e-8);
        double currentBeta = beta();
        double sizeD = static_cast<double>(size_);
        
        computeISWeights(
            tempPriorities_.data(),
            result.weights.data(),
            batchSize,
            totalP,
            sizeD,
            currentBeta
        );
        
        for (size_t i = 0; i < batchSize; ++i) {
            result.treeIndices[i] = static_cast<int32_t>(tempTreeIndices_[i]);
        }
        
        ++frame_;
        return result;
    }
    
    void updatePriorities(const int32_t* treeIndices, const double* tdErrors, size_t count) noexcept {
        double maxP = maxPriority_;
        
        for (size_t i = 0; i < count; ++i) {
            double priority = std::pow(std::abs(tdErrors[i]) + epsilon_, alpha_);
            tree_.update(static_cast<size_t>(treeIndices[i]), priority);
            maxP = std::max(maxP, priority);
        }
        
        maxPriority_ = maxP;
    }
    
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool isReady(size_t batchSize) const noexcept { return size_ >= batchSize; }

private:
    void computeISWeights(const double* priorities, float* weights, size_t count,
                          double totalP, double sizeD, double beta) const noexcept {
        double maxWeight = 0.0;
        
        for (size_t i = 0; i < count; ++i) {
            double prob = priorities[i] / totalP;
            double w = std::pow(sizeD * prob, -beta);
            weights[i] = static_cast<float>(w);
            maxWeight = std::max(maxWeight, w);
        }
        
        float invMax = 1.0f / static_cast<float>(maxWeight + 1e-8);
        
        size_t i = 0;
        #ifdef __AVX__
        __m256 invMaxVec = _mm256_set1_ps(invMax);
        for (; i + 8 <= count; i += 8) {
            __m256 w = _mm256_loadu_ps(weights + i);
            w = _mm256_mul_ps(w, invMaxVec);
            _mm256_storeu_ps(weights + i, w);
        }
        #endif
        
        for (; i < count; ++i) {
            weights[i] *= invMax;
        }
    }
    
    size_t capacity_;
    size_t stateDim_;
    size_t actionDim_;
    double alpha_;
    double betaStart_;
    size_t betaFrames_;
    double epsilon_;
    std::atomic<size_t> frame_;
    std::atomic<size_t> size_;
    std::atomic<double> maxPriority_;
    SumTree tree_;
    TransitionBlock data_;
    mutable std::mt19937 rng_;
    AlignedVector<size_t> tempTreeIndices_;
    AlignedVector<double> tempPriorities_;
    AlignedVector<size_t> tempDataIndices_;
};

class UniformReplayBuffer {
public:
    UniformReplayBuffer(size_t capacity, size_t stateDim, size_t actionDim)
        : capacity_(capacity),
          stateDim_(stateDim),
          actionDim_(actionDim),
          ptr_(0),
          size_(0),
          data_(capacity, stateDim, actionDim),
          rng_(std::random_device{}()) {}
    
    void push(const float* state, const float* action, float reward,
              const float* nextState, float done) noexcept {
        data_.store(ptr_, state, action, reward, nextState, done);
        ptr_ = (ptr_ + 1) % capacity_;
        size_ = std::min(size_ + 1, capacity_);
    }
    
    void pushBatch(const float* states, const float* actions, const float* rewards,
                   const float* nextStates, const float* dones, size_t batchSize) noexcept {
        data_.storeBatch(ptr_, states, actions, rewards, nextStates, dones, batchSize);
        ptr_ = (ptr_ + batchSize) % capacity_;
        size_ = std::min(size_ + batchSize, capacity_);
    }
    
    struct SampleResult {
        AlignedVector<float> states;
        AlignedVector<float> actions;
        AlignedVector<float> rewards;
        AlignedVector<float> nextStates;
        AlignedVector<float> dones;
    };
    
    SampleResult sample(size_t batchSize) {
        SampleResult result;
        result.states.resize(batchSize * stateDim_);
        result.actions.resize(batchSize * actionDim_);
        result.rewards.resize(batchSize);
        result.nextStates.resize(batchSize * stateDim_);
        result.dones.resize(batchSize);
        
        std::vector<size_t> indices(batchSize);
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);
        for (size_t i = 0; i < batchSize; ++i) {
            indices[i] = dist(rng_);
        }
        
        data_.gather(
            indices.data(), batchSize,
            result.states.data(),
            result.actions.data(),
            result.rewards.data(),
            result.nextStates.data(),
            result.dones.data()
        );
        
        return result;
    }
    
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool isReady(size_t batchSize) const noexcept { return size_ >= batchSize; }

private:
    size_t capacity_;
    size_t stateDim_;
    size_t actionDim_;
    size_t ptr_;
    size_t size_;
    TransitionBlock data_;
    mutable std::mt19937 rng_;
};

}
