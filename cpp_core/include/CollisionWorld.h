#pragma once
#include "Types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>

enum class ColliderType {
    SPHERE,
    AABB,
    CYLINDER,
    PLANE
};

struct AABB {
    Vec3 min;
    Vec3 max;
    
    AABB() noexcept : min(), max() {}
    AABB(const Vec3& min_, const Vec3& max_) noexcept : min(min_), max(max_) {}
    
    static AABB fromCenterSize(const Vec3& center, const Vec3& size) noexcept {
        Vec3 half = size * 0.5;
        return AABB(center - half, center + half);
    }
    
    Vec3 center() const noexcept {
        return (min + max) * 0.5;
    }
    
    Vec3 size() const noexcept {
        return max - min;
    }
    
    bool contains(const Vec3& point) const noexcept {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
    
    bool intersects(const AABB& other) const noexcept {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z;
    }
    
    double distanceTo(const Vec3& point) const noexcept {
        double dx = std::max({min.x - point.x, 0.0, point.x - max.x});
        double dy = std::max({min.y - point.y, 0.0, point.y - max.y});
        double dz = std::max({min.z - point.z, 0.0, point.z - max.z});
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    Vec3 closestPoint(const Vec3& point) const noexcept {
        return Vec3(
            std::clamp(point.x, min.x, max.x),
            std::clamp(point.y, min.y, max.y),
            std::clamp(point.z, min.z, max.z)
        );
    }
    
    AABB expanded(double margin) const noexcept {
        Vec3 m(margin, margin, margin);
        return AABB(min - m, max + m);
    }
};

struct Sphere {
    Vec3 center;
    double radius;
    
    Sphere() noexcept : center(), radius(0) {}
    Sphere(const Vec3& c, double r) noexcept : center(c), radius(r) {}
    
    bool contains(const Vec3& point) const noexcept {
        return (point - center).normSquared() <= radius * radius;
    }
    
    bool intersects(const Sphere& other) const noexcept {
        double dist_sq = (center - other.center).normSquared();
        double radii = radius + other.radius;
        return dist_sq <= radii * radii;
    }
    
    bool intersects(const AABB& box) const noexcept {
        Vec3 closest = box.closestPoint(center);
        return (closest - center).normSquared() <= radius * radius;
    }
    
    double distanceTo(const Vec3& point) const noexcept {
        return std::max(0.0, (point - center).norm() - radius);
    }
};

struct Cylinder {
    Vec3 base;
    double height;
    double radius;
    
    Cylinder() noexcept : base(), height(0), radius(0) {}
    Cylinder(const Vec3& b, double h, double r) noexcept : base(b), height(h), radius(r) {}
    
    bool contains(const Vec3& point) const noexcept {
        if (point.z < base.z || point.z > base.z + height) return false;
        double dx = point.x - base.x;
        double dy = point.y - base.y;
        return dx*dx + dy*dy <= radius * radius;
    }
    
    double distanceTo(const Vec3& point) const noexcept {
        double dx = point.x - base.x;
        double dy = point.y - base.y;
        double horiz_dist = std::max(0.0, std::sqrt(dx*dx + dy*dy) - radius);
        
        double vert_dist = 0.0;
        if (point.z < base.z) vert_dist = base.z - point.z;
        else if (point.z > base.z + height) vert_dist = point.z - base.z - height;
        
        return std::sqrt(horiz_dist*horiz_dist + vert_dist*vert_dist);
    }
};

struct Plane {
    Vec3 normal;
    double distance;
    
    Plane() noexcept : normal(0, 0, 1), distance(0) {}
    Plane(const Vec3& n, double d) noexcept : normal(n.normalized()), distance(d) {}
    
    static Plane fromPointNormal(const Vec3& point, const Vec3& normal) noexcept {
        Vec3 n = normal.normalized();
        return Plane(n, n.dot(point));
    }
    
    double signedDistanceTo(const Vec3& point) const noexcept {
        return normal.dot(point) - distance;
    }
    
    double distanceTo(const Vec3& point) const noexcept {
        return std::abs(signedDistanceTo(point));
    }
    
    bool isAbove(const Vec3& point) const noexcept {
        return signedDistanceTo(point) > 0;
    }
};

struct Collider {
    ColliderType type;
    int id;
    bool is_static;
    
    union {
        struct { Vec3 min; Vec3 max; } aabb;
        struct { Vec3 center; double radius; } sphere;
        struct { Vec3 base; double height; double radius; } cylinder;
        struct { Vec3 normal; double distance; } plane;
    };
    
    Collider() noexcept : type(ColliderType::SPHERE), id(-1), is_static(true) {
        sphere.center = Vec3();
        sphere.radius = 0;
    }
    
    static Collider createSphere(const Vec3& center, double radius, int id = -1) noexcept {
        Collider c;
        c.type = ColliderType::SPHERE;
        c.id = id;
        c.is_static = true;
        c.sphere.center = center;
        c.sphere.radius = radius;
        return c;
    }
    
    static Collider createAABB(const Vec3& min, const Vec3& max, int id = -1) noexcept {
        Collider c;
        c.type = ColliderType::AABB;
        c.id = id;
        c.is_static = true;
        c.aabb.min = min;
        c.aabb.max = max;
        return c;
    }
    
    static Collider createCylinder(const Vec3& base, double height, double radius, int id = -1) noexcept {
        Collider c;
        c.type = ColliderType::CYLINDER;
        c.id = id;
        c.is_static = true;
        c.cylinder.base = base;
        c.cylinder.height = height;
        c.cylinder.radius = radius;
        return c;
    }
    
    static Collider createPlane(const Vec3& normal, double distance, int id = -1) noexcept {
        Collider c;
        c.type = ColliderType::PLANE;
        c.id = id;
        c.is_static = true;
        c.plane.normal = normal.normalized();
        c.plane.distance = distance;
        return c;
    }
    
    double distanceTo(const Vec3& point) const noexcept {
        switch (type) {
            case ColliderType::SPHERE:
                return std::max(0.0, (point - sphere.center).norm() - sphere.radius);
            case ColliderType::AABB: {
                double dx = std::max({aabb.min.x - point.x, 0.0, point.x - aabb.max.x});
                double dy = std::max({aabb.min.y - point.y, 0.0, point.y - aabb.max.y});
                double dz = std::max({aabb.min.z - point.z, 0.0, point.z - aabb.max.z});
                return std::sqrt(dx*dx + dy*dy + dz*dz);
            }
            case ColliderType::CYLINDER: {
                double dx = point.x - cylinder.base.x;
                double dy = point.y - cylinder.base.y;
                double horiz = std::max(0.0, std::sqrt(dx*dx + dy*dy) - cylinder.radius);
                double vert = 0.0;
                if (point.z < cylinder.base.z) vert = cylinder.base.z - point.z;
                else if (point.z > cylinder.base.z + cylinder.height) 
                    vert = point.z - cylinder.base.z - cylinder.height;
                return std::sqrt(horiz*horiz + vert*vert);
            }
            case ColliderType::PLANE:
                return std::abs(plane.normal.dot(point) - plane.distance);
            default:
                return 0.0;
        }
    }
    
    bool contains(const Vec3& point, double margin = 0.0) const noexcept {
        return distanceTo(point) <= margin;
    }
};

struct CollisionResult {
    bool collision;
    int collider_id;
    double distance;
    Vec3 closest_point;
    Vec3 normal;
    double penetration;
    
    CollisionResult() noexcept 
        : collision(false), collider_id(-1), distance(0), 
          closest_point(), normal(), penetration(0) {}
};

struct RaycastResult {
    bool hit;
    int collider_id;
    double distance;
    Vec3 hit_point;
    Vec3 normal;
    
    RaycastResult() noexcept
        : hit(false), collider_id(-1), distance(0), hit_point(), normal() {}
};

class CollisionWorld {
public:
    static constexpr int MAX_COLLIDERS = 256;
    
    CollisionWorld() noexcept;
    
    int addCollider(const Collider& collider) noexcept;
    void removeCollider(int id) noexcept;
    void clearColliders() noexcept;
    
    void addSphere(const Vec3& center, double radius) noexcept;
    void addAABB(const Vec3& min, const Vec3& max) noexcept;
    void addCylinder(const Vec3& base, double height, double radius) noexcept;
    void addGroundPlane(double height = 0.0) noexcept;
    
    void addBuilding(const Vec3& position, const Vec3& size) noexcept;
    void addTree(const Vec3& position, double trunk_height, double trunk_radius, double canopy_radius) noexcept;
    void addPole(const Vec3& position, double height, double radius) noexcept;
    
    CollisionResult checkCollision(const Vec3& position, double radius) const noexcept;
    bool hasCollision(const Vec3& position, double radius) const noexcept;
    
    double distanceToNearest(const Vec3& position) const noexcept;
    int nearestColliderId(const Vec3& position) const noexcept;
    
    RaycastResult raycast(const Vec3& origin, const Vec3& direction, double max_distance = 1000.0) const noexcept;
    
    bool lineOfSight(const Vec3& from, const Vec3& to) const noexcept;
    
    std::vector<int> getCollidersInRadius(const Vec3& position, double radius) const noexcept;
    
    void generateRandomObstacles(int count, const AABB& bounds, double min_size, double max_size) noexcept;
    void generateUrbanEnvironment(const AABB& bounds, int building_count) noexcept;
    void generateForestEnvironment(const AABB& bounds, int tree_count) noexcept;
    
    int getColliderCount() const noexcept { return collider_count_; }
    const Collider& getCollider(int index) const noexcept { return colliders_[index]; }
    
    void reset() noexcept;
    
private:
    std::array<Collider, MAX_COLLIDERS> colliders_;
    int collider_count_ = 0;
    int next_id_ = 0;
    
    mutable std::mt19937 rng_;
    
    double rayIntersectSphere(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept;
    double rayIntersectAABB(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept;
    double rayIntersectPlane(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept;
};
