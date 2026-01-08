#include "CollisionWorld.h"
#include <random>

CollisionWorld::CollisionWorld() noexcept
    : colliders_(), collider_count_(0), next_id_(0)
    , rng_(std::random_device{}()) {}

int CollisionWorld::addCollider(const Collider& collider) noexcept {
    if (collider_count_ >= MAX_COLLIDERS) return -1;
    
    Collider c = collider;
    c.id = next_id_++;
    colliders_[collider_count_++] = c;
    return c.id;
}

void CollisionWorld::removeCollider(int id) noexcept {
    for (int i = 0; i < collider_count_; ++i) {
        if (colliders_[i].id == id) {
            colliders_[i] = colliders_[--collider_count_];
            return;
        }
    }
}

void CollisionWorld::clearColliders() noexcept {
    collider_count_ = 0;
}

void CollisionWorld::addSphere(const Vec3& center, double radius) noexcept {
    addCollider(Collider::createSphere(center, radius));
}

void CollisionWorld::addAABB(const Vec3& min, const Vec3& max) noexcept {
    addCollider(Collider::createAABB(min, max));
}

void CollisionWorld::addCylinder(const Vec3& base, double height, double radius) noexcept {
    addCollider(Collider::createCylinder(base, height, radius));
}

void CollisionWorld::addGroundPlane(double height) noexcept {
    addCollider(Collider::createPlane(Vec3(0, 0, 1), height));
}

void CollisionWorld::addBuilding(const Vec3& position, const Vec3& size) noexcept {
    Vec3 half = size * 0.5;
    Vec3 min(position.x - half.x, position.y - half.y, 0);
    Vec3 max(position.x + half.x, position.y + half.y, size.z);
    addAABB(min, max);
}

void CollisionWorld::addTree(const Vec3& position, double trunk_height, double trunk_radius, double canopy_radius) noexcept {
    addCylinder(position, trunk_height, trunk_radius);
    Vec3 canopy_center(position.x, position.y, position.z + trunk_height + canopy_radius * 0.5);
    addSphere(canopy_center, canopy_radius);
}

void CollisionWorld::addPole(const Vec3& position, double height, double radius) noexcept {
    addCylinder(position, height, radius);
}

CollisionResult CollisionWorld::checkCollision(const Vec3& position, double radius) const noexcept {
    CollisionResult result;
    result.distance = std::numeric_limits<double>::max();
    
    for (int i = 0; i < collider_count_; ++i) {
        const Collider& c = colliders_[i];
        double signedDist = 0.0;
        Vec3 normal;
        Vec3 closest;

        switch (c.type) {
            case ColliderType::SPHERE: {
                Vec3 d = position - c.sphere.center;
                double len = d.norm();
                if (len < 1e-9) {
                    normal = Vec3(0, 0, 1);
                    closest = c.sphere.center + normal * c.sphere.radius;
                    signedDist = -c.sphere.radius;
                } else {
                    normal = d * (1.0 / len);
                    closest = c.sphere.center + normal * c.sphere.radius;
                    signedDist = len - c.sphere.radius;
                }
                break;
            }
            case ColliderType::AABB: {
                Vec3 center = (c.aabb.min + c.aabb.max) * 0.5;
                Vec3 extents = (c.aabb.max - c.aabb.min) * 0.5;
                Vec3 d = position - center;
                Vec3 q(std::abs(d.x) - extents.x, std::abs(d.y) - extents.y, std::abs(d.z) - extents.z);
                
                double outsideFixed = Vec3(std::max(q.x, 0.0), std::max(q.y, 0.0), std::max(q.z, 0.0)).norm();
                double insideFixed = std::min(std::max({q.x, q.y, q.z}), 0.0);
                signedDist = insideFixed + outsideFixed;

                if (signedDist > 0) {
                    Vec3 clamped = position;
                    clamped.x = std::clamp(clamped.x, c.aabb.min.x, c.aabb.max.x);
                    clamped.y = std::clamp(clamped.y, c.aabb.min.y, c.aabb.max.y);
                    clamped.z = std::clamp(clamped.z, c.aabb.min.z, c.aabb.max.z);
                    closest = clamped;
                    Vec3 diff = position - closest;
                    normal = diff.normalized();
                } else {
                    Vec3 sign(d.x > 0 ? 1 : -1, d.y > 0 ? 1 : -1, d.z > 0 ? 1 : -1);
                    if (q.x > q.y && q.x > q.z) {
                        normal = Vec3(sign.x, 0, 0);
                        closest = Vec3(center.x + extents.x * sign.x, position.y, position.z);
                    } else if (q.y > q.z) {
                        normal = Vec3(0, sign.y, 0);
                        closest = Vec3(position.x, center.y + extents.y * sign.y, position.z);
                    } else {
                        normal = Vec3(0, 0, sign.z);
                        closest = Vec3(position.x, position.y, center.z + extents.z * sign.z);
                    }
                }
                break;
            }
            case ColliderType::CYLINDER: {
                Vec2 d(position.x - c.cylinder.base.x, position.y - c.cylinder.base.y);
                double distH = d.norm();
                
                double z = position.z - c.cylinder.base.z;
                double halfH = c.cylinder.height * 0.5;
                double centeredZ = z - halfH; 
                
                double dH = distH - c.cylinder.radius;
                double dV = std::abs(centeredZ) - halfH;
                
                double outsideFixed = Vec2(std::max(dH, 0.0), std::max(dV, 0.0)).norm();
                double insideFixed = std::min(std::max(dH, dV), 0.0);
                signedDist = insideFixed + outsideFixed;
                
                if (signedDist > 0) {
                     if (dH > 0 && dV > 0) {
                        Vec2 dirH = d.normalized();
                        closest = Vec3(
                            c.cylinder.base.x + dirH.x * c.cylinder.radius,
                            c.cylinder.base.y + dirH.y * c.cylinder.radius,
                            c.cylinder.base.z + halfH + (centeredZ > 0 ? halfH : -halfH)
                        );
                        normal = (position - closest).normalized();
                     } else if (dH > 0) {
                        Vec2 dirH = d.normalized();
                        closest = Vec3(
                             c.cylinder.base.x + dirH.x * c.cylinder.radius,
                             c.cylinder.base.y + dirH.y * c.cylinder.radius,
                             position.z
                        );
                        normal = Vec3(dirH.x, dirH.y, 0);
                     } else {
                         closest = Vec3(position.x, position.y, c.cylinder.base.z + halfH + (centeredZ > 0 ? halfH : -halfH));
                         normal = Vec3(0, 0, centeredZ > 0 ? 1.0 : -1.0);
                     }
                } else {
                    if (dH > dV) {
                        Vec2 dirH = d.normalized();
                        normal = Vec3(dirH.x, dirH.y, 0);
                        closest = Vec3(
                             c.cylinder.base.x + dirH.x * c.cylinder.radius,
                             c.cylinder.base.y + dirH.y * c.cylinder.radius,
                             position.z
                        );
                    } else {
                        normal = Vec3(0, 0, centeredZ > 0 ? 1.0 : -1.0);
                        closest = Vec3(position.x, position.y, c.cylinder.base.z + halfH + (centeredZ > 0 ? halfH : -halfH));
                    }
                }
                break;
            }
            case ColliderType::PLANE: {
                signedDist = c.plane.normal.dot(position) - c.plane.distance;
                normal = c.plane.normal;
                closest = position - normal * signedDist;
                break;
            }
        }
        
        if (signedDist < radius) {
            double currentDist = (signedDist < 0) ? signedDist : signedDist; 
            if (std::abs(currentDist) < std::abs(result.distance)) {
                 result.distance = signedDist;
                 result.collider_id = c.id;
                 result.collision = true;
                 result.normal = normal;
                 result.closest_point = closest;
                 result.penetration = radius - signedDist;
            }
        }
    }
    
    return result;
}

bool CollisionWorld::hasCollision(const Vec3& position, double radius) const noexcept {
    CollisionResult res = checkCollision(position, radius);
    return res.collision;
}

double CollisionWorld::distanceToNearest(const Vec3& position) const noexcept {
    double minDist = std::numeric_limits<double>::max();
    for(int i=0; i<collider_count_; ++i) {
        double d = colliders_[i].distanceTo(position);
        if(d < minDist) minDist = d;
    }
    return minDist;
}

int CollisionWorld::nearestColliderId(const Vec3& position) const noexcept {
    double minDist = std::numeric_limits<double>::max();
    int id = -1;
    for(int i=0; i<collider_count_; ++i) {
        double d = colliders_[i].distanceTo(position);
        if(d < minDist) {
            minDist = d;
            id = colliders_[i].id;
        }
    }
    return id;
}

double CollisionWorld::rayIntersectSphere(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept {
    Vec3 oc = origin - c.sphere.center;
    double a = dir.dot(dir);
    double b = 2.0 * oc.dot(dir);
    double cc = oc.dot(oc) - c.sphere.radius * c.sphere.radius;
    double discriminant = b*b - 4*a*cc;
    
    if (discriminant < 0) return -1.0;
    
    double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
    return t > 0 ? t : (-b + std::sqrt(discriminant)) / (2.0 * a);
}

double CollisionWorld::rayIntersectAABB(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept {
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax = std::numeric_limits<double>::infinity();
    
    double dirs[3] = {dir.x, dir.y, dir.z};
    double origs[3] = {origin.x, origin.y, origin.z};
    double mins[3] = {c.aabb.min.x, c.aabb.min.y, c.aabb.min.z};
    double maxs[3] = {c.aabb.max.x, c.aabb.max.y, c.aabb.max.z};
    
    for (int i = 0; i < 3; ++i) {
        if (std::abs(dirs[i]) < 1e-12) {
            if (origs[i] < mins[i] || origs[i] > maxs[i]) return -1.0;
        } else {
            double t1 = (mins[i] - origs[i]) / dirs[i];
            double t2 = (maxs[i] - origs[i]) / dirs[i];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return -1.0;
        }
    }
    
    return tmin > 0 ? tmin : tmax;
}

double CollisionWorld::rayIntersectPlane(const Vec3& origin, const Vec3& dir, const Collider& c) const noexcept {
    double denom = c.plane.normal.dot(dir);
    if (std::abs(denom) < 1e-12) return -1.0;
    
    double t = (c.plane.distance - c.plane.normal.dot(origin)) / denom;
    return t > 0 ? t : -1.0;
}

RaycastResult CollisionWorld::raycast(const Vec3& origin, const Vec3& direction, double max_distance) const noexcept {
    RaycastResult result;
    result.distance = max_distance;
    
    Vec3 dir = direction.normalized();
    
    for (int i = 0; i < collider_count_; ++i) {
        double t = -1.0;
        
        switch (colliders_[i].type) {
            case ColliderType::SPHERE:
                t = rayIntersectSphere(origin, dir, colliders_[i]);
                break;
            case ColliderType::AABB:
                t = rayIntersectAABB(origin, dir, colliders_[i]);
                break;
            case ColliderType::PLANE:
                t = rayIntersectPlane(origin, dir, colliders_[i]);
                break;
            default:
                break;
        }
        
        if (t > 0 && t < result.distance) {
            result.hit = true;
            result.distance = t;
            result.collider_id = colliders_[i].id;
            result.hit_point = origin + dir * t;
            
            switch (colliders_[i].type) {
                case ColliderType::SPHERE: {
                    Vec3 diff = result.hit_point - colliders_[i].sphere.center;
                    result.normal = diff.normalized();
                    break;
                }
                case ColliderType::AABB: {
                    Vec3 center = (colliders_[i].aabb.min + colliders_[i].aabb.max) * 0.5;
                    Vec3 diff = result.hit_point - center;
                    Vec3 size = colliders_[i].aabb.max - colliders_[i].aabb.min;
                    Vec3 n;
                    double max_comp = 0;
                    for (int j = 0; j < 3; ++j) {
                        double comp[3] = {diff.x/size.x, diff.y/size.y, diff.z/size.z};
                        if (std::abs(comp[j]) > max_comp) {
                            max_comp = std::abs(comp[j]);
                            n = Vec3(j==0?1:0, j==1?1:0, j==2?1:0) * (comp[j] > 0 ? 1 : -1);
                        }
                    }
                    result.normal = n;
                    break;
                }
                case ColliderType::PLANE:
                    result.normal = colliders_[i].plane.normal;
                    break;
                default:
                    result.normal = Vec3(0, 0, 1);
            }
        }
    }
    
    return result;
}

bool CollisionWorld::lineOfSight(const Vec3& from, const Vec3& to) const noexcept {
    Vec3 diff = to - from;
    double dist = diff.norm();
    if (dist < 1e-6) return true;
    
    RaycastResult result = raycast(from, diff, dist);
    return !result.hit || result.distance >= dist - 1e-6;
}

std::vector<int> CollisionWorld::getCollidersInRadius(const Vec3& position, double radius) const noexcept {
    std::vector<int> result;
    for (int i = 0; i < collider_count_; ++i) {
        if (colliders_[i].distanceTo(position) <= radius) {
            result.push_back(colliders_[i].id);
        }
    }
    return result;
}

void CollisionWorld::generateRandomObstacles(int count, const AABB& bounds, double min_size, double max_size) noexcept {
    std::uniform_real_distribution<double> dist_x(bounds.min.x, bounds.max.x);
    std::uniform_real_distribution<double> dist_y(bounds.min.y, bounds.max.y);
    std::uniform_real_distribution<double> dist_size(min_size, max_size);
    std::uniform_int_distribution<int> dist_type(0, 2);
    
    for (int i = 0; i < count && collider_count_ < MAX_COLLIDERS; ++i) {
        double x = dist_x(rng_);
        double y = dist_y(rng_);
        double size = dist_size(rng_);
        
        int type = dist_type(rng_);
        switch (type) {
            case 0:
                addSphere(Vec3(x, y, size), size);
                break;
            case 1:
                addAABB(Vec3(x - size, y - size, 0), Vec3(x + size, y + size, size * 2));
                break;
            case 2:
                addCylinder(Vec3(x, y, 0), size * 3, size * 0.5);
                break;
        }
    }
}

void CollisionWorld::generateUrbanEnvironment(const AABB& bounds, int building_count) noexcept {
    addGroundPlane(0);
    
    std::uniform_real_distribution<double> dist_x(bounds.min.x + 5, bounds.max.x - 5);
    std::uniform_real_distribution<double> dist_y(bounds.min.y + 5, bounds.max.y - 5);
    std::uniform_real_distribution<double> dist_height(10, 50);
    std::uniform_real_distribution<double> dist_width(5, 15);
    
    for (int i = 0; i < building_count && collider_count_ < MAX_COLLIDERS - 10; ++i) {
        double x = dist_x(rng_);
        double y = dist_y(rng_);
        double height = dist_height(rng_);
        double width = dist_width(rng_);
        double depth = dist_width(rng_);
        
        addBuilding(Vec3(x, y, 0), Vec3(width, depth, height));
    }
}

void CollisionWorld::generateForestEnvironment(const AABB& bounds, int tree_count) noexcept {
    addGroundPlane(0);
    
    std::uniform_real_distribution<double> dist_x(bounds.min.x + 1, bounds.max.x - 1);
    std::uniform_real_distribution<double> dist_y(bounds.min.y + 1, bounds.max.y - 1);
    std::uniform_real_distribution<double> dist_height(3, 10);
    std::uniform_real_distribution<double> dist_trunk(0.1, 0.3);
    std::uniform_real_distribution<double> dist_canopy(1, 3);
    
    for (int i = 0; i < tree_count && collider_count_ < MAX_COLLIDERS - 10; ++i) {
        double x = dist_x(rng_);
        double y = dist_y(rng_);
        double trunk_height = dist_height(rng_);
        double trunk_radius = dist_trunk(rng_);
        double canopy_radius = dist_canopy(rng_);
        
        addTree(Vec3(x, y, 0), trunk_height, trunk_radius, canopy_radius);
    }
}

void CollisionWorld::reset() noexcept {
    clearColliders();
    next_id_ = 0;
}
