// The rocker differential, enforced.
//
//     ROCKER DIFFERENTIAL, THE THING THAT MAKES IT A ROCKER-BOGIE
//     theta_body = (theta_rocker_left + theta_rocker_right) / 2
//
// Build spec section 07. The two rockers are coupled through a differential
// bar so the body pitch is their average. Skip it and you have modelled a
// rigid six-wheeler that happens to look like a rocker-bogie, and every
// obstacle result will be wrong.
//
// The URDF also declares an SDFormat gearbox joint. Not every physics engine
// implements one, and a constraint that silently does nothing is worse than
// no constraint, because gate 2 would pass for the wrong reason. This plugin
// is the version that always runs, and the phase 3 gate checks the residual
// in simulation rather than trusting either form.
//
// How it holds:
//
//   Take the absolute rocker pitches to be body pitch plus the joint angle.
//   Requiring the body to bisect them reduces, exactly, to
//
//       C = theta_left + theta_right = 0
//
//   which is derived symbolically in tools/derive_symbolic.py. Both joints
//   have unit sensitivity to C, so the corrective generalised force is the
//   same on each: a stiff spring and damper on the violation. That is also
//   what a real differential bar does, through its own compliance, which is
//   why this is a physical model rather than a numerical trick.

#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/Name.hh>
#include <gz/common/Console.hh>

#include <string>
#include <vector>

namespace ares
{
namespace sim
{

class RockerDifferential
  : public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
    const gz::sim::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    gz::sim::EntityComponentManager & ecm,
    gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(entity);
    if (!this->model.Valid(ecm)) {
      gzerr << "[ares_differential] not attached to a model, doing nothing.\n";
      return;
    }

    this->leftName = sdf->Get<std::string>("left_joint", "rocker_left_joint").first;
    this->rightName = sdf->Get<std::string>("right_joint", "rocker_right_joint").first;

    // Stiff enough that the residual stays in the noise, soft enough that a
    // one millisecond step stays stable. The phase 3 gate measures the
    // residual rather than assuming these are right.
    this->stiffness = sdf->Get<double>("stiffness", 4000.0).first;
    this->damping = sdf->Get<double>("damping", 120.0).first;

    this->left = this->model.JointByName(ecm, this->leftName);
    this->right = this->model.JointByName(ecm, this->rightName);

    if (this->left == gz::sim::kNullEntity || this->right == gz::sim::kNullEntity) {
      gzerr << "[ares_differential] could not find joints '" << this->leftName
            << "' and '" << this->rightName
            << "'. The differential is NOT being enforced. Every obstacle "
               "result from this run is suspect.\n";
      return;
    }

    this->ready = true;
    gzmsg << "[ares_differential] coupling " << this->leftName << " and "
          << this->rightName << ", stiffness " << this->stiffness
          << ", damping " << this->damping << "\n";
  }

  void PreUpdate(
    const gz::sim::UpdateInfo & info,
    gz::sim::EntityComponentManager & ecm) override
  {
    if (!this->ready || info.paused) {
      return;
    }

    this->Ensure<gz::sim::components::JointPosition>(ecm, this->left);
    this->Ensure<gz::sim::components::JointPosition>(ecm, this->right);
    this->Ensure<gz::sim::components::JointVelocity>(ecm, this->left);
    this->Ensure<gz::sim::components::JointVelocity>(ecm, this->right);

    const auto * leftPos = ecm.Component<gz::sim::components::JointPosition>(this->left);
    const auto * rightPos = ecm.Component<gz::sim::components::JointPosition>(this->right);
    const auto * leftVel = ecm.Component<gz::sim::components::JointVelocity>(this->left);
    const auto * rightVel = ecm.Component<gz::sim::components::JointVelocity>(this->right);

    if (!leftPos || !rightPos || !leftVel || !rightVel ||
      leftPos->Data().empty() || rightPos->Data().empty() ||
      leftVel->Data().empty() || rightVel->Data().empty())
    {
      return;
    }

    // C = theta_left + theta_right, and its rate.
    const double violation = leftPos->Data()[0] + rightPos->Data()[0];
    const double rate = leftVel->Data()[0] + rightVel->Data()[0];

    // Both joints have unit sensitivity to C, so both take the same force.
    const double correction = -this->stiffness * violation - this->damping * rate;

    this->AddForce(ecm, this->left, correction);
    this->AddForce(ecm, this->right, correction);

    this->residual = violation;
  }

  /// The constraint violation, in radians. Zero is a satisfied differential.
  double Residual() const {return this->residual;}

private:
  template<typename T>
  void Ensure(gz::sim::EntityComponentManager & ecm, gz::sim::Entity joint)
  {
    if (!ecm.Component<T>(joint)) {
      ecm.CreateComponent(joint, T());
    }
  }

  void AddForce(
    gz::sim::EntityComponentManager & ecm, gz::sim::Entity joint, double force)
  {
    auto * cmd = ecm.Component<gz::sim::components::JointForceCmd>(joint);
    if (!cmd) {
      ecm.CreateComponent(
        joint, gz::sim::components::JointForceCmd({force}));
      return;
    }
    if (cmd->Data().empty()) {
      cmd->Data().push_back(force);
    } else {
      // Add rather than assign. A drive controller may already have written
      // a command to this joint, and clobbering it would make the
      // differential silently disable the drive.
      cmd->Data()[0] += force;
    }
  }

  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Entity left{gz::sim::kNullEntity};
  gz::sim::Entity right{gz::sim::kNullEntity};
  std::string leftName;
  std::string rightName;
  double stiffness{4000.0};
  double damping{120.0};
  double residual{0.0};
  bool ready{false};
};

}  // namespace sim
}  // namespace ares

GZ_ADD_PLUGIN(
  ares::sim::RockerDifferential,
  gz::sim::System,
  ares::sim::RockerDifferential::ISystemConfigure,
  ares::sim::RockerDifferential::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ares::sim::RockerDifferential, "ares::sim::RockerDifferential")
