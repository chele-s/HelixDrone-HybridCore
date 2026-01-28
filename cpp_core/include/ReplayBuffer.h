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

struct alignas(64) SequenceBlock {
    float* observations;
    float* next_observations;
    float* actions;
    float* rewards;
    float* dones;
    int64_t* episode_ids;
    size_t obs_dim;
    size_t action_dim;
    size_t capacity;
    
    SequenceBlock(size_t cap, size_t oDim, size_t aDim) 
        : obs_dim(oDim), action_dim(aDim), capacity(cap) {
        size_t obsBytes = ((cap * oDim * sizeof(float)) + 63) & ~63;
        size_t actionBytes = ((cap * aDim * sizeof(float)) + 63) & ~63;
        size_t scalarBytes = ((cap * sizeof(float)) + 63) & ~63;
        size_t idBytes = ((cap * sizeof(int64_t)) + 63) & ~63;
        
        observations = static_cast<float*>(ALIGNED_ALLOC(64, obsBytes));
        next_observations = static_cast<float*>(ALIGNED_ALLOC(64, obsBytes));
        actions = static_cast<float*>(ALIGNED_ALLOC(64, actionBytes));
        rewards = static_cast<float*>(ALIGNED_ALLOC(64, scalarBytes));
        dones = static_cast<float*>(ALIGNED_ALLOC(64, scalarBytes));
        episode_ids = static_cast<int64_t*>(ALIGNED_ALLOC(64, idBytes));
        
        std::memset(observations, 0, obsBytes);
        std::memset(next_observations, 0, obsBytes);
        std::memset(actions, 0, actionBytes);
        std::memset(rewards, 0, scalarBytes);
        std::memset(dones, 0, scalarBytes);
        std::memset(episode_ids, 0, idBytes);
    }
    
    ~SequenceBlock() {
        ALIGNED_FREE(observations);
        ALIGNED_FREE(next_observations);
        ALIGNED_FREE(actions);
        ALIGNED_FREE(rewards);
        ALIGNED_FREE(dones);
        ALIGNED_FREE(episode_ids);
    }
    
    SequenceBlock(const SequenceBlock&) = delete;
    SequenceBlock& operator=(const SequenceBlock&) = delete;
    
    void store(size_t idx, const float* obs, const float* act, float rew,
               const float* next_obs, float done, int64_t ep_id) noexcept {
        std::memcpy(observations + idx * obs_dim, obs, obs_dim * sizeof(float));
        std::memcpy(actions + idx * action_dim, act, action_dim * sizeof(float));
        std::memcpy(next_observations + idx * obs_dim, next_obs, obs_dim * sizeof(float));
        rewards[idx] = rew;
        dones[idx] = done;
        episode_ids[idx] = ep_id;
    }
    
    void gatherSequences(const size_t* start_indices, size_t batch_size, size_t seq_len,
                         float* out_obs, float* out_next_obs, float* out_actions,
                         float* out_rewards, float* out_dones) const noexcept {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t t = 0; t < seq_len; ++t) {
                size_t idx = (start_indices[b] + t) % capacity;
                size_t out_idx = b * seq_len + t;
                std::memcpy(out_obs + out_idx * obs_dim, observations + idx * obs_dim, obs_dim * sizeof(float));
                std::memcpy(out_next_obs + out_idx * obs_dim, next_observations + idx * obs_dim, obs_dim * sizeof(float));
                std::memcpy(out_actions + out_idx * action_dim, actions + idx * action_dim, action_dim * sizeof(float));
                out_rewards[out_idx] = rewards[idx];
                out_dones[out_idx] = dones[idx];
            }
        }
    }
};

class SequenceReplayBuffer {
public:
    struct SampleResult {
        AlignedVector<float> obs_seq;
        AlignedVector<float> next_obs_seq;
        AlignedVector<float> action_seq;
        AlignedVector<float> rewards;
        AlignedVector<float> dones;
        AlignedVector<float> masks;
        AlignedVector<size_t> start_indices;
        AlignedVector<float> burn_in_obs;
        AlignedVector<float> burn_in_next_obs;
        AlignedVector<float> burn_in_actions;
        AlignedVector<float> burn_in_masks;
        size_t batch_size;
        size_t seq_len;
        size_t burn_in_len;
        size_t obs_dim;
        size_t action_dim;
    };
    
    SequenceReplayBuffer(size_t capacity, size_t obs_dim, size_t action_dim, size_t seq_len, size_t burn_in_len = 0)
        : capacity_(capacity),
          obs_dim_(obs_dim),
          action_dim_(action_dim),
          seq_len_(seq_len),
          burn_in_len_(burn_in_len),
          total_seq_len_(seq_len + burn_in_len),
          ptr_(0),
          size_(0),
          valid_count_(0),
          current_episode_id_(0),
          data_(capacity, obs_dim, action_dim),
          rng_(std::random_device{}()),
          valid_mask_((capacity + 63) / 64, 0) {}
    
    void push(const float* obs, const float* action, float reward, 
              const float* next_obs, float done) noexcept {
        data_.store(ptr_, obs, action, reward, next_obs, done, current_episode_id_);
        updateValidity(ptr_);
        ptr_ = (ptr_ + 1) % capacity_;
        size_ = std::min(size_ + 1, capacity_);
        if (done > 0.5f) ++current_episode_id_;
    }
    
    void pushBatch(const float* obs, const float* actions, const float* rewards,
                   const float* next_obs, const float* dones, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            push(obs + i * obs_dim_, actions + i * action_dim_, rewards[i],
                 next_obs + i * obs_dim_, dones[i]);
        }
    }
    
    SampleResult sample(size_t batch_size) {
        SampleResult result;
        result.batch_size = batch_size;
        result.seq_len = seq_len_;
        result.burn_in_len = burn_in_len_;
        result.obs_dim = obs_dim_;
        result.action_dim = action_dim_;
        
        size_t total_steps = batch_size * total_seq_len_;
        AlignedVector<float> full_obs(total_steps * obs_dim_);
        AlignedVector<float> full_next_obs(total_steps * obs_dim_);
        AlignedVector<float> full_actions(total_steps * action_dim_);
        AlignedVector<float> full_rewards(total_steps);
        AlignedVector<float> full_dones(total_steps);
        AlignedVector<float> full_masks(total_steps, 1.0f);
        
        result.start_indices.resize(batch_size);
        
        if (valid_count_ == 0) {
            result.obs_seq.resize(batch_size * seq_len_ * obs_dim_, 0.0f);
            result.next_obs_seq.resize(batch_size * seq_len_ * obs_dim_, 0.0f);
            result.action_seq.resize(batch_size * seq_len_ * action_dim_, 0.0f);
            result.rewards.resize(batch_size, 0.0f);
            result.dones.resize(batch_size, 0.0f);
            result.masks.resize(batch_size * seq_len_, 0.0f);
            if (burn_in_len_ > 0) {
                result.burn_in_obs.resize(batch_size * burn_in_len_ * obs_dim_, 0.0f);
                result.burn_in_next_obs.resize(batch_size * burn_in_len_ * obs_dim_, 0.0f);
                result.burn_in_actions.resize(batch_size * burn_in_len_ * action_dim_, 0.0f);
                result.burn_in_masks.resize(batch_size * burn_in_len_, 0.0f);
            }
            return result;
        }
        
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);
        size_t collected = 0;
        size_t max_attempts = batch_size * 100;
        size_t attempts = 0;
        
        while (collected < batch_size && attempts < max_attempts) {
            size_t idx = dist(rng_);
            if (checkBit(idx)) {
                result.start_indices[collected] = idx;
                ++collected;
            }
            ++attempts;
        }
        
        if (collected < batch_size) {
            for (size_t i = collected; i < batch_size; ++i) {
                result.start_indices[i] = result.start_indices[i % std::max(collected, size_t(1))];
            }
        }
        
        data_.gatherSequences(result.start_indices.data(), batch_size, total_seq_len_,
                              full_obs.data(), full_next_obs.data(),
                              full_actions.data(), full_rewards.data(), full_dones.data());
        
        size_t train_steps = batch_size * seq_len_;
        result.obs_seq.resize(train_steps * obs_dim_);
        result.next_obs_seq.resize(train_steps * obs_dim_);
        result.action_seq.resize(train_steps * action_dim_);
        result.rewards.resize(batch_size);
        result.dones.resize(batch_size);
        result.masks.resize(train_steps, 1.0f);
        
        if (burn_in_len_ > 0) {
            size_t burn_steps = batch_size * burn_in_len_;
            result.burn_in_obs.resize(burn_steps * obs_dim_);
            result.burn_in_next_obs.resize(burn_steps * obs_dim_);
            result.burn_in_actions.resize(burn_steps * action_dim_);
            result.burn_in_masks.resize(burn_steps, 1.0f);
        }
        
        for (size_t b = 0; b < batch_size; ++b) {
            if (burn_in_len_ > 0) {
                for (size_t t = 0; t < burn_in_len_; ++t) {
                    size_t src_idx = b * total_seq_len_ + t;
                    size_t dst_idx = b * burn_in_len_ + t;
                    std::memcpy(result.burn_in_obs.data() + dst_idx * obs_dim_,
                               full_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                    std::memcpy(result.burn_in_next_obs.data() + dst_idx * obs_dim_,
                               full_next_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                    std::memcpy(result.burn_in_actions.data() + dst_idx * action_dim_,
                               full_actions.data() + src_idx * action_dim_, action_dim_ * sizeof(float));
                    result.burn_in_masks[dst_idx] = full_masks[src_idx];
                }
            }
            
            for (size_t t = 0; t < seq_len_; ++t) {
                size_t src_idx = b * total_seq_len_ + burn_in_len_ + t;
                size_t dst_idx = b * seq_len_ + t;
                std::memcpy(result.obs_seq.data() + dst_idx * obs_dim_,
                           full_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                std::memcpy(result.next_obs_seq.data() + dst_idx * obs_dim_,
                           full_next_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                std::memcpy(result.action_seq.data() + dst_idx * action_dim_,
                           full_actions.data() + src_idx * action_dim_, action_dim_ * sizeof(float));
                result.masks[dst_idx] = full_masks[src_idx];
            }
            
            size_t last_full_idx = b * total_seq_len_ + total_seq_len_ - 1;
            result.rewards[b] = full_rewards[last_full_idx];
            result.dones[b] = full_dones[last_full_idx];
        }
        
        return result;
    }
    
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t burnInLength() const noexcept { return burn_in_len_; }
    bool isReady(size_t batch_size) const noexcept { return valid_count_ >= batch_size; }

private:
    void updateValidity(size_t ptr) noexcept {
        for (size_t offset = 0; offset < total_seq_len_; ++offset) {
            size_t start = (ptr + capacity_ - offset) % capacity_;
            if (checkBit(start)) {
                clearBit(start);
                --valid_count_;
            }
        }
        
        if (size_ >= total_seq_len_) {
            size_t candidate = (ptr + capacity_ - total_seq_len_ + 1) % capacity_;
            if (isValidSequence(candidate) && !checkBit(candidate)) {
                setBit(candidate);
                ++valid_count_;
            }
        }
    }
    
    bool isValidSequence(size_t start) const noexcept {
        size_t end = (start + total_seq_len_ - 1) % capacity_;
        if (data_.episode_ids[start] != data_.episode_ids[end]) return false;
        for (size_t t = 0; t < total_seq_len_ - 1; ++t) {
            size_t idx = (start + t) % capacity_;
            if (data_.dones[idx] > 0.5f) return false;
        }
        return true;
    }
    
    bool checkBit(size_t idx) const noexcept { 
        return (valid_mask_[idx / 64] & (1ULL << (idx % 64))) != 0; 
    }
    void setBit(size_t idx) noexcept { valid_mask_[idx / 64] |= (1ULL << (idx % 64)); }
    void clearBit(size_t idx) noexcept { valid_mask_[idx / 64] &= ~(1ULL << (idx % 64)); }
    
    size_t capacity_, obs_dim_, action_dim_, seq_len_, burn_in_len_, total_seq_len_;
    size_t ptr_, size_, valid_count_;
    int64_t current_episode_id_;
    SequenceBlock data_;
    mutable std::mt19937 rng_;
    AlignedVector<uint64_t> valid_mask_;
};

class SequencePrioritizedReplayBuffer {
public:
    struct SampleResult {
        AlignedVector<float> obs_seq;
        AlignedVector<float> next_obs_seq;
        AlignedVector<float> action_seq;
        AlignedVector<float> rewards;
        AlignedVector<float> dones;
        AlignedVector<float> masks;
        AlignedVector<float> weights;
        AlignedVector<size_t> start_indices;
        AlignedVector<float> burn_in_obs;
        AlignedVector<float> burn_in_next_obs;
        AlignedVector<float> burn_in_actions;
        AlignedVector<float> burn_in_masks;
        size_t batch_size;
        size_t seq_len;
        size_t burn_in_len;
        size_t obs_dim;
        size_t action_dim;
    };
    
    SequencePrioritizedReplayBuffer(size_t capacity, size_t obs_dim, size_t action_dim, 
                                     size_t seq_len, size_t burn_in_len = 0, double alpha = 0.6,
                                     double beta_start = 0.4, size_t beta_frames = 100000,
                                     double epsilon = 1e-6)
        : capacity_(capacity),
          obs_dim_(obs_dim),
          action_dim_(action_dim),
          seq_len_(seq_len),
          burn_in_len_(burn_in_len),
          total_seq_len_(seq_len + burn_in_len),
          alpha_(alpha),
          beta_start_(beta_start),
          beta_frames_(beta_frames),
          epsilon_(epsilon),
          frame_(1),
          ptr_(0),
          size_(0),
          valid_count_(0),
          current_episode_id_(0),
          max_priority_(1.0),
          data_(capacity, obs_dim, action_dim),
          priorities_(capacity, 0.0),
          rng_(std::random_device{}()),
          valid_mask_((capacity + 63) / 64, 0) {}
    
    double beta() const noexcept {
        double progress = static_cast<double>(frame_) / static_cast<double>(beta_frames_);
        return std::min(1.0, beta_start_ + (1.0 - beta_start_) * progress);
    }
    
    void push(const float* obs, const float* action, float reward,
              const float* next_obs, float done) noexcept {
        data_.store(ptr_, obs, action, reward, next_obs, done, current_episode_id_);
        priorities_[ptr_] = std::pow(max_priority_, alpha_);
        updateValidity(ptr_);
        ptr_ = (ptr_ + 1) % capacity_;
        size_ = std::min(size_ + 1, capacity_);
        if (done > 0.5f) ++current_episode_id_;
    }
    
    void pushBatch(const float* obs, const float* actions, const float* rewards,
                   const float* next_obs, const float* dones, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            push(obs + i * obs_dim_, actions + i * action_dim_, rewards[i],
                 next_obs + i * obs_dim_, dones[i]);
        }
    }
    
    SampleResult sample(size_t batch_size) {
        SampleResult result;
        result.batch_size = batch_size;
        result.seq_len = seq_len_;
        result.burn_in_len = burn_in_len_;
        result.obs_dim = obs_dim_;
        result.action_dim = action_dim_;
        
        size_t total_steps = batch_size * total_seq_len_;
        AlignedVector<float> full_obs(total_steps * obs_dim_);
        AlignedVector<float> full_next_obs(total_steps * obs_dim_);
        AlignedVector<float> full_actions(total_steps * action_dim_);
        AlignedVector<float> full_rewards(total_steps);
        AlignedVector<float> full_dones(total_steps);
        AlignedVector<float> full_masks(total_steps, 1.0f);
        
        result.start_indices.resize(batch_size);
        result.weights.resize(batch_size);
        
        if (valid_count_ == 0) {
            result.obs_seq.resize(batch_size * seq_len_ * obs_dim_, 0.0f);
            result.next_obs_seq.resize(batch_size * seq_len_ * obs_dim_, 0.0f);
            result.action_seq.resize(batch_size * seq_len_ * action_dim_, 0.0f);
            result.rewards.resize(batch_size, 0.0f);
            result.dones.resize(batch_size, 0.0f);
            result.masks.resize(batch_size * seq_len_, 0.0f);
            if (burn_in_len_ > 0) {
                result.burn_in_obs.resize(batch_size * burn_in_len_ * obs_dim_, 0.0f);
                result.burn_in_next_obs.resize(batch_size * burn_in_len_ * obs_dim_, 0.0f);
                result.burn_in_actions.resize(batch_size * burn_in_len_ * action_dim_, 0.0f);
                result.burn_in_masks.resize(batch_size * burn_in_len_, 0.0f);
            }
            return result;
        }
        
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);
        
        size_t collected = 0;
        size_t max_attempts = batch_size * 50;
        size_t attempts = 0;
        
        while (collected < batch_size && attempts < max_attempts) {
            size_t idx = dist(rng_);
            if (checkBit(idx)) {
                result.start_indices[collected] = idx;
                ++collected;
            }
            ++attempts;
        }
        
        if (collected < batch_size) {
            for (size_t i = collected; i < batch_size; ++i) {
                size_t src = i % std::max(collected, size_t(1));
                result.start_indices[i] = result.start_indices[src];
            }
        }
        
        double current_beta = beta();
        double max_weight = 0.0;
        double total_priority = 0.0;
        AlignedVector<double> selected_priorities(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            selected_priorities[i] = computeSeqPriority(result.start_indices[i]);
            total_priority += selected_priorities[i];
        }
        if (total_priority < 1e-10) total_priority = 1.0;
        
        for (size_t i = 0; i < batch_size; ++i) {
            double prob = selected_priorities[i] / total_priority;
            double w = std::pow(static_cast<double>(size_) * prob, -current_beta);
            result.weights[i] = static_cast<float>(w);
            max_weight = std::max(max_weight, w);
        }
        
        float inv_max = 1.0f / static_cast<float>(max_weight + 1e-10);
        for (size_t i = 0; i < batch_size; ++i) {
            result.weights[i] *= inv_max;
        }
        
        data_.gatherSequences(result.start_indices.data(), batch_size, total_seq_len_,
                              full_obs.data(), full_next_obs.data(),
                              full_actions.data(), full_rewards.data(), full_dones.data());
        
        size_t train_steps = batch_size * seq_len_;
        result.obs_seq.resize(train_steps * obs_dim_);
        result.next_obs_seq.resize(train_steps * obs_dim_);
        result.action_seq.resize(train_steps * action_dim_);
        result.rewards.resize(batch_size);
        result.dones.resize(batch_size);
        result.masks.resize(train_steps, 1.0f);
        
        if (burn_in_len_ > 0) {
            size_t burn_steps = batch_size * burn_in_len_;
            result.burn_in_obs.resize(burn_steps * obs_dim_);
            result.burn_in_next_obs.resize(burn_steps * obs_dim_);
            result.burn_in_actions.resize(burn_steps * action_dim_);
            result.burn_in_masks.resize(burn_steps, 1.0f);
        }
        
        for (size_t b = 0; b < batch_size; ++b) {
            if (burn_in_len_ > 0) {
                for (size_t t = 0; t < burn_in_len_; ++t) {
                    size_t src_idx = b * total_seq_len_ + t;
                    size_t dst_idx = b * burn_in_len_ + t;
                    std::memcpy(result.burn_in_obs.data() + dst_idx * obs_dim_,
                               full_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                    std::memcpy(result.burn_in_next_obs.data() + dst_idx * obs_dim_,
                               full_next_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                    std::memcpy(result.burn_in_actions.data() + dst_idx * action_dim_,
                               full_actions.data() + src_idx * action_dim_, action_dim_ * sizeof(float));
                    result.burn_in_masks[dst_idx] = full_masks[src_idx];
                }
            }
            
            for (size_t t = 0; t < seq_len_; ++t) {
                size_t src_idx = b * total_seq_len_ + burn_in_len_ + t;
                size_t dst_idx = b * seq_len_ + t;
                std::memcpy(result.obs_seq.data() + dst_idx * obs_dim_,
                           full_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                std::memcpy(result.next_obs_seq.data() + dst_idx * obs_dim_,
                           full_next_obs.data() + src_idx * obs_dim_, obs_dim_ * sizeof(float));
                std::memcpy(result.action_seq.data() + dst_idx * action_dim_,
                           full_actions.data() + src_idx * action_dim_, action_dim_ * sizeof(float));
                result.masks[dst_idx] = full_masks[src_idx];
            }
            
            size_t last_full_idx = b * total_seq_len_ + total_seq_len_ - 1;
            result.rewards[b] = full_rewards[last_full_idx];
            result.dones[b] = full_dones[last_full_idx];
        }
        
        ++frame_;
        return result;
    }
    
    void updatePriorities(const size_t* indices, const double* td_errors, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            double priority = std::pow(std::abs(td_errors[i]) + epsilon_, alpha_);
            for (size_t t = 0; t < total_seq_len_; ++t) {
                priorities_[(indices[i] + t) % capacity_] = priority;
            }
            double current_max = max_priority_.load();
            while (priority > current_max && !max_priority_.compare_exchange_weak(current_max, priority));
        }
    }
    
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t burnInLength() const noexcept { return burn_in_len_; }
    bool isReady(size_t batch_size) const noexcept { return valid_count_ >= batch_size; }

private:
    void updateValidity(size_t ptr) noexcept {
        for (size_t offset = 0; offset < total_seq_len_; ++offset) {
            size_t start = (ptr + capacity_ - offset) % capacity_;
            if (checkBit(start)) {
                clearBit(start);
                --valid_count_;
            }
        }
        if (size_ >= total_seq_len_) {
            size_t candidate = (ptr + capacity_ - total_seq_len_ + 1) % capacity_;
            if (isValidSequence(candidate) && !checkBit(candidate)) {
                setBit(candidate);
                ++valid_count_;
            }
        }
    }
    
    bool isValidSequence(size_t start) const noexcept {
        size_t end = (start + total_seq_len_ - 1) % capacity_;
        if (data_.episode_ids[start] != data_.episode_ids[end]) return false;
        for (size_t t = 0; t < total_seq_len_ - 1; ++t) {
            size_t idx = (start + t) % capacity_;
            if (data_.dones[idx] > 0.5f) return false;
        }
        return true;
    }
    
    double computeSeqPriority(size_t start) const noexcept {
        double p = 0.0;
        for (size_t t = 0; t < total_seq_len_; ++t) {
            p += priorities_[(start + t) % capacity_];
        }
        return p / static_cast<double>(total_seq_len_);
    }
    
    double computeMaxSeqPriority() const noexcept {
        return std::pow(max_priority_.load(), alpha_);
    }
    
    bool checkBit(size_t idx) const noexcept { 
        return (valid_mask_[idx / 64] & (1ULL << (idx % 64))) != 0; 
    }
    void setBit(size_t idx) noexcept { valid_mask_[idx / 64] |= (1ULL << (idx % 64)); }
    void clearBit(size_t idx) noexcept { valid_mask_[idx / 64] &= ~(1ULL << (idx % 64)); }
    
    size_t capacity_, obs_dim_, action_dim_, seq_len_, burn_in_len_, total_seq_len_;
    double alpha_, beta_start_, epsilon_;
    size_t beta_frames_;
    std::atomic<size_t> frame_;
    size_t ptr_, size_, valid_count_;
    int64_t current_episode_id_;
    std::atomic<double> max_priority_;
    SequenceBlock data_;
    AlignedVector<double> priorities_;
    mutable std::mt19937 rng_;
    AlignedVector<uint64_t> valid_mask_;
};

}
