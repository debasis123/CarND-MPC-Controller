/*               
* @Author: Udacity
* @Last Modified by:   debasis123
*/

#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

class MPC {
public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  std::vector<double> Solve(const Eigen::VectorXd& state, const Eigen::VectorXd& coeffs);
};

#endif /* MPC_H */
