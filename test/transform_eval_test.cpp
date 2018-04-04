#include <iostream>
#include <Eigen/Dense>

int main() {

  auto *translation = new double[3]{6.2,4.3,6.9};
  auto *center = new double[3]{1.0,2.0,3.3};
  auto *rotation = new double[4]{8.0,2.0,2.6,9.8};
  auto *scaleOrientation = new double[4]{3.0,3.0,2.2,2.5};
  auto *scale = new double[3]{2.0,1.2,3.5};


  Eigen::Matrix4d T;
  T<<1.0,0.0,0.0,translation[0],0.0,1.0,0.0,translation[1],0.0,0.0,1.0,translation[2],0.0,0.0,0.0,1.0;

  Eigen::Matrix4d C;
  C<<1.0,0.0,0.0,center[0],0.0,1.0,0.0,center[1],0.0,0.0,1.0,center[2],0.0,0.0,0.0,1.0;

  Eigen::Vector3d k;
  k << rotation[0], rotation[1], rotation[2];
  double ang = rotation[3];
  double kmag = k.norm();

  if (kmag < 1e-9) { // Can't directly compare doubles.
    kmag = 1.0; // null rotation
    k << 0.0, 0.0, 1.0;
    ang = 0.0;
  }

  k /= kmag;

  Eigen::Matrix3d RK; // Cross product matrix
  RK<<0.0,-k[2],k[1],k[2],0.0,-k[0],-k[1],k[0],0.0;

  Eigen::Matrix3d eye;
  eye << 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0;
  Eigen::Matrix<double,3,1> Right;
  Right << 0.0,0.0,0.0;
  Eigen::Matrix<double,1,4> Bottom;
  Bottom << 0.0,0.0,0.0,1.0;

  // RTopLeft is the top left 3x3 double matrix inside of R
  Eigen::Matrix3d RTopLeft = eye.array() + (sin(ang) * RK).array() + ((1.0 - cos(ang)) * (RK * RK)).array();

  Eigen::Matrix4d R(RTopLeft.rows()+Bottom.rows(),RTopLeft.cols()+Right.cols());
  R << RTopLeft, Right, Bottom;

  // Apply Rodrigues rotation formula to determine scale orientation
  Eigen::Vector3d SOk;
  SOk << scaleOrientation[0], scaleOrientation[1], scaleOrientation[2];
  double SOang = scaleOrientation[3];
  double SOkmag = SOk.norm();

  if (SOkmag < 1e-9) { // Can't directly compare doubles.
    SOkmag = 1.0; // null rotation
    SOk << 0.0, 0.0, 1.0;
    SOang = 0.0;
  }

  SOk/=SOkmag;

  Eigen::Matrix3d SOK; // Cross product matrix
  SOK<<0.0,-SOk[2],SOk[1],SOk[2],0.0,-SOk[0],-SOk[1],SOk[0],0.0;

  // SRTopLeft is the top left 3x3 double matrix inside of SR
  Eigen::Matrix3d SRTopLeft = eye.array() + (sin(SOang) * SOK).array() + ((1.0 - cos(SOang)) * (SOK * SOK)).array();

  Eigen::Matrix4d SR(SRTopLeft.rows()+Bottom.rows(),SRTopLeft.cols()+Right.cols());
  SR << SRTopLeft, Right, Bottom;

  Eigen::Matrix4d S;
  S << scale[0], 0.0, 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 0.0, scale[2], 0.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::Matrix4d matrix;
  matrix = T * C * R * SR * S * (-SR) * (-C);

  std::cout << matrix << std::endl;

  delete[] translation;
  delete[] center;
  delete[] rotation;
  delete[] scaleOrientation;
  delete[] scale;

  return 0;
}