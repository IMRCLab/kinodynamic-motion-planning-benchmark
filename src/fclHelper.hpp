#pragma once

#include <fcl/fcl.h>

template <typename S>
void shiftAABB(fcl::AABB<S>& aabb, const fcl::Vector3<S>& offset)
{
  aabb.min_ += offset;
  aabb.max_ += offset;
}

template <typename S>
class ShiftableDynamicAABBTreeCollisionManager : public fcl::DynamicAABBTreeCollisionManager<S>
{
public:
  void shift(const fcl::Vector3<S> &offset)
  {
    // Use const-cast, since DynamicAABBTreeCollisionManager has tree as a private member
    const auto& tree = this->getTree();
    // std::cout << "b: " << tree.getRoot()->bv.min_ << std::endl;
    shift_recursive(const_cast<fcl::detail::NodeBase<fcl::AABB<S>> *>(tree.getRoot()), offset);
    // std::cout << "a: " << tree.getRoot()->bv.min_ << std::endl;
    // std::cout << "s: " << this->size() << std::endl;
  }

protected:
  void shift_recursive(fcl::detail::NodeBase<fcl::AABB<S>> *node, const fcl::Vector3<S> &offset)
  {
    if (node == nullptr) {
      return;
    }
    shiftAABB(node->bv, offset);
    if (node->isLeaf()) {
      fcl::CollisionObject<S> *obj = static_cast<fcl::CollisionObject<S> *>(node->data);
      if (!obj->isFree()) {
        obj->setTranslation(obj->getTranslation() + offset);
        obj->computeAABB();
        assert(node->bv.equal(obj->getAABB()));
        // std::cout << "ot" << std::endl;
      }
    } else {
      shift_recursive(node->children[0], offset);
      shift_recursive(node->children[1], offset);
    }
  }
};