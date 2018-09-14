/*               
* @Author: Udacity
* @Last Modified by:   debasis123
*/

#include "MPC.h"
#include <vector>
#include <cmath>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

using CppAD::AD;

// TODO: Set the timestep length and duration
// Balancing between the computational cost and effectiveness
const size_t N = 25;
const double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Both the reference cross track and orientation errors are 0.
const double ref_cte  = 0;
const double ref_epsi = 0;
const double ref_v    = 65;

// The lpopt solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should establish
// when one variable starts and another ends to make our life easier.
const size_t x_start     = 0;
const size_t y_start     = x_start + N;
const size_t psi_start   = y_start + N;
const size_t v_start     = psi_start + N;
const size_t cte_start   = v_start + N;
const size_t epsi_start  = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start     = delta_start + N - 1;


/**
 * Class that holds input to the ipopt module
 * and computes objective function and constraints
 */
class FG_eval {
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs_;

public:
  FG_eval(const Eigen::VectorXd& coeffs) {
    this->coeffs_ = coeffs;
  }

  using ADvector = CPPAD_TESTVECTOR(AD<double>);

  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and the Solver function below.

    // The cost is stored in the first element of `fg`.
    // 0 is the index at whichIpopt expects fg to store the cost value
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // The part of the cost based on the reference state.
    // Define the cost related the reference state and add anything you think may be beneficial.

    // The penalty weights below are obtained by their relative importance
    // and running the simulator multiple times

    // penalties for drifting from the reference
    // penalize heavily for cte and epsi, 
    int w_cte = 2500, w_epsi = 2500, w_vel = 10;
    
    // minimize our cross track, heading, and velocity errors
    for (size_t t = 0; t != N; ++t) {
      fg[0] += w_cte  * CppAD::pow(vars[cte_start+t] - ref_cte, 2);
      fg[0] += w_epsi * CppAD::pow(vars[epsi_start+t] - ref_epsi, 2);
      fg[0] += w_vel  * CppAD::pow(vars[v_start+t] - ref_v, 2);
    }

    // Minimize the use of actuators: constrain erratic control inputs
    // For example, if we're making a turn, we'd like the turn to be smooth, not sharp. 
    // Additionally, the vehicle velocity should not change too radically.

    // penalize moderately for sharp turns or sudden acceleration / break
    int w_delta = 200, w_a = 200, w_vel_steer = 2000;
    
    for (size_t t = 0; t != N-1; ++t) {
      fg[0] += w_delta * CppAD::pow(vars[delta_start+t], 2);
      fg[0] += w_a * CppAD::pow(vars[a_start+t], 2);
      // penalty for speed + steer, from another student
      fg[0] += w_vel_steer * CppAD::pow(vars[delta_start+t] * vars[v_start+t], 2);
    }

    // Minimize the value gap between sequential actuations.
    // make control decisions more consistent, or smoother.
    // The next control input should be similar to the current one.

    // penalize slightly for changes in delta and a in successive timestamps
    int w_delta_dt = 10, w_a_dt = 10;
    
    for (size_t t = 0; t != N-2; ++t) {
      fg[0] += w_delta_dt * CppAD::pow(vars[delta_start+t+1] - vars[delta_start+t], 2);
      fg[0] += w_a_dt     * CppAD::pow(vars[a_start+t+1] - vars[a_start+t], 2);
    }
    
    // Setup Model Constraints

    // Initial constraints
    // We add 1 to each of the starting indices due to cost being located at index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1+x_start]    = vars[x_start];
    fg[1+y_start]    = vars[y_start];
    fg[1+psi_start]  = vars[psi_start];
    fg[1+v_start]    = vars[v_start];
    fg[1+cte_start]  = vars[cte_start];
    fg[1+epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (size_t t = 1; t != N; ++t) {
      // The state at time t+1 .
      AD<double> x1    = vars[x_start+t];
      AD<double> y1    = vars[y_start+t];
      AD<double> psi1  = vars[psi_start+t];
      AD<double> v1    = vars[v_start+t];
      AD<double> cte1  = vars[cte_start+t];
      AD<double> epsi1 = vars[epsi_start+t];

      // The state at time t.
      AD<double> x0    = vars[x_start+t-1];
      AD<double> y0    = vars[y_start+t-1];
      AD<double> psi0  = vars[psi_start+t-1];
      AD<double> v0    = vars[v_start+t-1];
      AD<double> cte0  = vars[cte_start+t-1];
      AD<double> epsi0 = vars[epsi_start+t-1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start+t-1];
      AD<double> a0     = vars[a_start+t-1];

      // the reference trajectory is fit by a 3rd order polynomial
      AD<double> f0       = coeffs_[0] + coeffs_[1]*x0 + coeffs_[2]*CppAD::pow(x0, 2) + coeffs_[3]*CppAD::pow(x0, 3);
      AD<double> psi_des0 = CppAD::atan(coeffs_[1] + 2*coeffs_[2]*x0 + 3*coeffs_[3]*CppAD::pow(x0, 2));

      // the 6 equations to represent MPC model
      fg[1+x_start+t]    = x1 - (x0 + v0*CppAD::cos(psi0)*dt);
      fg[1+y_start+t]    = y1 - (y0 + v0*CppAD::sin(psi0)*dt);
      // to deal with the simulator which thinks positive delta means right turn
      // we change the + to - before delta0
      fg[1+psi_start+t]  = psi1 - (psi0 - delta0*(v0/Lf)*dt);
      fg[1+v_start+t]    = v1 - (v0 + a0*dt);
      fg[1+cte_start+t]  = cte1 - ((f0 - y0) + (v0*CppAD::sin(epsi0)*dt));
      // same for the following equation as well
      fg[1+epsi_start+t] = epsi1 - ((psi0 - psi_des0) - delta0*(v0/Lf)*dt);
    }
  }  
};


//////////////////////////
// MPC class definition //
//////////////////////////

MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(const Eigen::VectorXd& state, const Eigen::VectorXd& coeffs) {
  using Dvector = CPPAD_TESTVECTOR(double);

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  const size_t n_vars = N*6 + (N-1)*2;
  // TODO: Set the number of constraints
  const size_t n_constraints = N*6;

  // Initial value of the independent variables. SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i != n_vars; ++i) {
    vars[i] = 0.0;
  }

  // Set the initial variable values
  const auto x    = state(0);
  const auto y    = state(1);
  const auto psi  = state(2);
  const auto v    = state(3);
  const auto cte  = state(4);
  const auto epsi = state(5);

  vars[x_start]    = x;
  vars[y_start]    = y;
  vars[psi_start]  = psi;
  vars[v_start]    = v;
  vars[cte_start]  = cte;
  vars[epsi_start] = epsi;

  // Lower and upper limits for x
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  
  // TODO: Set lower and upper limits for variables.

  // Set all non-actuators upper and lowerlimits to the max negative and positive values.
  for (size_t i = 0; i != delta_start; ++i) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25 degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (size_t i = delta_start; i != a_start; ++i) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (size_t i = a_start; i != n_vars; ++i) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i != n_constraints; ++i) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start]    = x;
  constraints_lowerbound[y_start]    = y;
  constraints_lowerbound[psi_start]  = psi;
  constraints_lowerbound[v_start]    = v;
  constraints_lowerbound[cte_start]  = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start]    = x;
  constraints_upperbound[y_start]    = y;
  constraints_upperbound[psi_start]  = psi;
  constraints_upperbound[v_start]    = v;
  constraints_upperbound[cte_start]  = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // NOTE: You don't have to worry about these options
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this if your N*dt is large
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, 
      vars, vars_lowerbound, vars_upperbound, 
      constraints_lowerbound, constraints_upperbound, 
      fg_eval, 
      solution);

  // Check some of the solution values
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  // std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with `solution.x[i]`.
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  // We return x and y predicted values as well to draw them on the simulator
  std::vector<double> values;

  values.push_back(solution.x[delta_start]);
  values.push_back(solution.x[a_start]);
  
  for (size_t i=1; i!=N; ++i) {
    values.push_back(solution.x[x_start+i]);
    values.push_back(solution.x[y_start+i]);
  }

  return values;
}
