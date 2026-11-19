// TinyDEM
// Copyright (c) Roman Vetter
// vetterro@ethz.ch
// ETH Zurich

#ifndef TINYDEM_HPP
#define TINYDEM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <random>
#include <sstream>
#include <string>
#include <vector>

typedef double real; // floating point precision

constexpr real PI = 3.141592653589793238462643383279502884L; // portable definition of pi
constexpr real big = std::numeric_limits<real>::max(); // largest representable number
constexpr std::size_t invalid = std::numeric_limits<std::size_t>::max(); // invalid index

// minimal class providing the required vector algebra in 3D
struct Point
{
  real x, y, z;
  Point operator+(const Point& p) const { return {x + p.x, y + p.y, z + p.z}; } // sum
  Point operator-(const Point& p) const { return {x - p.x, y - p.y, z - p.z}; } // difference
  real operator*(const Point& p) const { return x * p.x + y * p.y + z * p.z; } // dot product
  Point cross(const Point& p) const { return {y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x}; } // cross product
  real length() const { return std::sqrt(x * x + y * y + z * z); } // norm
  void add(const real a, const Point& p) // add a * p to this point with atomic updates
  {
    #pragma omp atomic
    x += a * p.x;
    #pragma omp atomic
    y += a * p.y;
    #pragma omp atomic
    z += a * p.z;
  }
};
inline Point operator*(const real a, const Point& p) { return {a * p.x, a * p.y, a * p.z}; } // scalar vector multiplication

// minimal class providing the required bounding box functionality
struct Box
{
  Point min, max; // lower and upper corners
  constexpr Box(const Point& p0 = {big,big,big}, const Point& p1 = {-big,-big,-big}) : min(p0), max(p1) {}
  void include(const Point& p0, const Point& p1) // assuming p1 >= p0
  {
    min = {std::min(min.x, p0.x), std::min(min.y, p0.y), std::min(min.z, p0.z)};
    max = {std::max(max.x, p1.x), std::max(max.y, p1.y), std::max(max.z, p1.z)};
  }
  bool contains(const Point& p) const
  {
    return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z;
  }
};

// minimal class providing the required quaternion algebra
struct Quaternion
{
  real s; // scalar part
  Point v; // vectorial part
  void left_mult_exp(const Point& p) // left-multiply by the quaternion exponential of p
  {
    const real p_norm = p.length();
    const real c = std::cos(p_norm);
    const Point u = p_norm > 0 ? std::sin(p_norm) / p_norm * p : p; // vectorial part of exponential
    *this = {c * s - u * v, c * v + s * u + u.cross(v)};
  }
  void update(const real dt, const Point& w, const Point& alpha) // SPIRAL quaternion update
  {
    left_mult_exp(dt / 2 * w);
    left_mult_exp(dt * dt / 4 * alpha);
  }
};

// data struct for contacts
struct Contact
{
  std::size_t j; // index of the other object
  Point us, ur; // sliding and rolling friction spring displacements
  real ut; // twisting spring displacement
  bool active; // flag used to remove contacts that have ended
};

// minimal particle struct
struct Particle
{
  real R; // radius
  Point x, v, a, w, alpha; // position, velocity, acceleration, angular velocity, angular acceleration
  Quaternion q; // orientation
  std::size_t next; // next particle in same cell
  std::list<Contact> c; // list of contact data for collisions
};

// minimal spatial cell class
struct Cell
{
  std::ptrdiff_t x, y, z; // signed indices in x, y, z directions
  explicit Cell(const Point& p = {0,0,0}, const real d = 1) :
    x(static_cast<std::ptrdiff_t>(p.x / d)),
    y(static_cast<std::ptrdiff_t>(p.y / d)),
    z(static_cast<std::ptrdiff_t>(p.z / d)) {}
};

// minimal mesh face struct
struct Face
{
  std::vector<std::size_t> v; // vertices
  real R; // semi-thickness
  Box box; // bounding box
};

// returns the closest point on the line segment (a,b) to point p
inline Point edge_point(const Point& p, const Point& a, const Point& b)
{
  const Point ab = b - a, ap = p - a;
  const real s = ab * ap / (ab * ab);
  if (s <= 0) return a; // closest point is vertex a
  if (s >= 1) return b; // closest point is vertex b
  return a + s * ab; // closest point is between a and b
}

// returns the closest point on the triangle (a,b,c) to point p
inline Point triangle_point(const Point& p, const Point& a, const Point& b, const Point& c)
{
  const Point ab = b - a, ac = c - a, ap = p - a;
  const real d1 = ab * ap, d2 = ac * ap;
  if (d1 <= 0 && d2 <= 0) return a; // closest point is vertex a

  const Point bp = p - b;
  const real d3 = ab * bp, d4 = ac * bp;
  if (d3 >= 0 && d4 <= d3) return b; // closest point is vertex b

  const real vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + (d1 / (d1 - d3)) * ab; // closest point is on ab edge

  const Point cp = p - c;
  const real d5 = ab * cp, d6 = ac * cp;
  if (d6 >= 0 && d5 <= d6) return c; // closest point is vertex c

  const real vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + (d2 / (d2 - d6)) * ac; // closest point is on ac edge

  const real va = d3 * d6 - d5 * d4, d43 = d4 - d3, d56 = d5 - d6;
  if (va <= 0 && d43 >= 0 && d56 >= 0) return b + (d43 / (d43 + d56)) * (c - b); // closest point is on bc edge

  const real invv = 1 / (va + vb + vc);
  return a + (vb * invv) * ab + (vc * invv) * ac; // closest point is in interior
}

// returns the closest point on the rectangle (a,b,c,d=b+c-a) to point p
inline Point rectangle_point(const Point& p, const Point& a, const Point& b, const Point& c)
{
  return edge_point(p, a, b) + edge_point(p, a, c) - a;
}

// minimal class for polygonal meshes in the Object File Format
struct Mesh
{
  std::vector<Point> p; // list of points
  std::vector<Face> f; // list of faces

  // read a mesh from an OFF file
  template <std::size_t NR>
  Mesh(const std::string& name, const real (&Rmesh)[NR])
  {
    std::ifstream file(name);
    if (!file.is_open()) return;

    readline(file); // skip header line "OFF"
    std::size_t Np = 0, Nf = 0; // number of points and faces
    std::istringstream(readline(file)) >> Np >> Nf;

    // read points
    p.resize(Np);
    for (auto& point : p)
      std::istringstream(readline(file)) >> point.x >> point.y >> point.z;

    // read faces
    f.reserve(Nf);
    for (std::size_t i = 0; i < Nf; ++i)
    {
      std::istringstream iss(readline(file));
      std::size_t Nv = 0; // number of vertices of the face
      if (!(iss >> Nv) || Nv < 1 || Nv > 4) continue; // skip faces that are neither point, edge, triangle nor rectangle

      Face face;
      face.v.resize(Nv);
      bool ok = true;
      for (auto& v : face.v)
        ok = (iss >> v) && v < p.size() && ok; // each vertex must be present and in range
      std::size_t index = 0;
      iss >> index; // optional Rmesh index, defaults to 0
      if (!ok || index >= NR) continue; // skip faces with invalid vertex indices
      face.R = Rmesh[index];

      // precompute bounding box
      const Point pad = {face.R, face.R, face.R}; // padding by the semi-thickness
      for (const auto& v : face.v)
        face.box.include(p[v] - pad, p[v] + pad);

      f.push_back(face);
    }
  }

  // read a line from a stream, skipping empty and comment-only lines
  static std::string readline(std::istream& s)
  {
    std::string line;
    std::size_t i;
    while (std::getline(s, line))
      if ((i = line.find_first_not_of(" \t\r")) != std::string::npos && line[i] != '#')
        return line; // first line with content that isn't a comment
    return line; // end of stream: getline has cleared the string, so this is empty
  }

  // returns the point on face j that is closest to x
  Point closest_point(const Point& x, std::size_t j) const
  {
    const auto& v = f[j].v;
    switch (v.size())
    {
      case 1 : return p[v[0]]; // point
      case 2 : return edge_point(x, p[v[0]], p[v[1]]); // edge
      case 3 : return triangle_point(x, p[v[0]], p[v[1]], p[v[2]]); // triangle
      default: return rectangle_point(x, p[v[0]], p[v[1]], p[v[3]]); // rectangle
    }
  }
};

// ensemble of particles
struct Ensemble
{
  std::vector<Particle> p; // list of particles
  const Mesh& mesh;
  real t = 0; // time
  std::mt19937 rng; // random number generator
  std::uniform_real_distribution<real> uni_dist; // standard uniform distribution U[0,1)
  real rn = uni_dist(rng); // uniform random number for the next particle
  std::vector<std::size_t> head; // head of particle list in each cell
  Cell Nc; // number of cells in each direction

  // read a particle ensemble from a CSV file
  Ensemble(const std::string& name, const Mesh& m) : mesh(m)
  {
    std::ifstream file(name);
    if (!file.is_open()) return;
    std::string line;
    std::getline(file, line); // skip header line
    while (std::getline(file, line))
    {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream iss(line);
      Particle P{}; // value-initialized: a, alpha and the contact list start out empty/zero
      if (iss >> P.x.x >> P.x.y >> P.x.z >> P.R >> P.v.x >> P.v.y >> P.v.z
              >> P.q.s >> P.q.v.x >> P.q.v.y >> P.q.v.z >> P.w.x >> P.w.y >> P.w.z)
        p.push_back(P);
    }
  }

  // first particle in cell
  std::size_t& first(const Cell& c) { return head[static_cast<std::size_t>((c.x * Nc.y + c.y) * Nc.z + c.z)]; }

  // call f() for every particle in the cell range [cmin-1, cmax+1]
  template <class F>
  void for_nearby(const Cell& cmin, const Cell& cmax, const F& f)
  {
    Cell c;
    for (c.x = std::max<std::ptrdiff_t>(cmin.x, 1) - 1; c.x <= cmax.x + 1 && c.x < Nc.x; ++c.x)
      for (c.y = std::max<std::ptrdiff_t>(cmin.y, 1) - 1; c.y <= cmax.y + 1 && c.y < Nc.y; ++c.y)
        for (c.z = std::max<std::ptrdiff_t>(cmin.z, 1) - 1; c.z <= cmax.z + 1 && c.z < Nc.z; ++c.z)
          for (std::size_t j = first(c); j != invalid; j = p[j].next)
            f(j);
  }

  void step(); // defined externally

  // print & write output
  void output(const std::string& outdir, const std::size_t f) const
  {
    // print frame number, simulated time, and number of particles to stdout
    std::cout << "frame " << f << ": t=" << t << ", " << p.size() << " particles" << std::endl;

    // write current particle state to a CSV file
    std::ostringstream name;
    name << "particles" << std::setw(6) << std::setfill('0') << f << ".csv";
    std::ofstream file(outdir + "/" + name.str());
    file << std::setprecision(std::numeric_limits<real>::max_digits10);
    file << "x,y,z,R,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz\n";
    for (const auto& P : p)
      file << P.x.x << "," << P.x.y << "," << P.x.z << "," << P.R << ","
           << P.v.x << "," << P.v.y << "," << P.v.z << ","
           << P.q.s << "," << P.q.v.x << "," << P.q.v.y << "," << P.q.v.z << ","
           << P.w.x << "," << P.w.y << "," << P.w.z << "\n";
  }
};

#endif // TINYDEM_HPP
