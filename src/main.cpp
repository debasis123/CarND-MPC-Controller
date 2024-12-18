/*               
* @Author: Udacity
* @Last Modified by:   debasis123
*/

#include <cmath>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() {
  return M_PI; 
}
double deg2rad(const double x) {
  return x * pi() / 180; 
}
double rad2deg(const double x) {
  return x * 180 / pi(); 
}

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2-b1+2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(const Eigen::VectorXd& coeffs, const double x) {
  double result = 0.0;
  for (size_t i = 0; i != coeffs.size(); ++i) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(const Eigen::VectorXd& xvals,
                        const Eigen::VectorXd& yvals,
                        const int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size()-1);

  Eigen::MatrixXd A(xvals.size(), order+1); // rows x col

  // first col set to 1.0 each
  for (size_t row = 0; row != xvals.size(); ++row) {
    A(row, 0) = 1.0;
  }

  // rest of the cols
  for (size_t row = 0; row != xvals.size(); ++row) {
    for (size_t col = 0; col != order; ++col) {
      A(row, col+1) = A(row, col)*xvals(row);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);

  return result;
}


int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    // cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"]; // The global x positions of the waypoints
          vector<double> ptsy = j[1]["ptsy"]; // The global y positions of the waypoints.
          assert(ptsx.size() == ptsy.size());
          double px           = j[1]["x"]; // The global x position of the vehicle.
          double py           = j[1]["y"]; // The global y position of the vehicle.
          double psi          = j[1]["psi"]; // The orientation of the vehicle in radians
          double v            = j[1]["speed"]; // The current velocity in mph
          
          // move the co-ordinate system from global to the car
          // the calculation involving the polynomial becomes easy 
          // 
          // first, shift the frame of reference to the car, that is, (px, py) becomes (0, 0)
          for (size_t i = 0; i != ptsx.size(); ++i) {
            double shift_x = ptsx.at(i) - px;
            double shift_y = ptsy.at(i) - py;
            // now rotate car axis around the origin to make it 0
            ptsx.at(i) = shift_x*cos(0-psi) - shift_y*sin(0-psi);
            ptsy.at(i) = shift_x*sin(0-psi) + shift_y*cos(0-psi);
          }

          // we need to express the location vectors in terms of Eigen vectors
          // as the polyfit function expects that
          double* ptrx = &ptsx[0];
          Eigen::Map<Eigen::VectorXd> ptsx_transform(ptrx, 6);
          double* ptry = &ptsy[0];
          Eigen::Map<Eigen::VectorXd> ptsy_transform(ptry, 6);

          // We fit a 3rd order polynomial to the above points
          auto coeffs_pred_traj = polyfit(ptsx_transform, ptsy_transform, 3);

          // calculate cte & orientation error
          // Note that our car is at the origin (0, 0) and the orientation of it is 0 as well
          // ideally, the cte should be the perpendicular distance between the polynomial and origin
          double cte  = 0/*=py*/ - polyeval(coeffs_pred_traj, 0/*=px*/);
          double epsi = 0/*=psi*/ - atan(coeffs_pred_traj[1]) /* + 2*0*coeffs[2] + 3*coeffs[3]*pow(0,2)* */;

          Eigen::VectorXd state(6);
          state << 0/*=px*/, 0/*=py*/, 0/*=psi*/, v, cte, epsi;

          auto vars = mpc.Solve(state, coeffs_pred_traj);

          // TODO: Calculate steering angle and throttle using MPC.
          // Both are in between [-1, 1].
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          double steer_value    = vars.at(0) / deg2rad(25);
          double throttle_value = vars.at(1);

          // send back the following information to the simulator
          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;


          ///////////////////
          // VISUALIZATION //
          ///////////////////

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          for (size_t i = 2; i != vars.size(); ++i) {
            (i%2 == 0) ? mpc_x_vals.push_back(vars[i])
                       : mpc_y_vals.push_back(vars[i]);
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints / reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          const double poly_inc = 2.5;
          const size_t nPoints = 25;
          for (size_t i = 1; i != nPoints; ++i){
            next_x_vals.push_back(poly_inc*i);
            next_y_vals.push_back(polyeval(coeffs_pred_traj, poly_inc*i));
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be able to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
