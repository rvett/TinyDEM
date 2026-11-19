// TinyDEM
// Copyright (c) Roman Vetter
// vetterro@ethz.ch
// ETH Zurich

#include "tinydem.hpp"

constexpr real Rmin = 0.002; // [L] minimum particle radius
constexpr real Rmax = 0.01; // [L] maximum particle radius
constexpr real Rmesh [] = {0.002,0.02,0.01}; // [L] list of mesh radii (length must match number of indices in mesh file)
constexpr Box source = {{-0.1,-0.1,0.3}, {0.1,0.1,0.4}}; // [L] source region: particles spawn with their center inside
constexpr Box sink; // [L] sink region: particles with their center inside are removed (empty by default)

constexpr real rho = 1e3; // [M/L^3] mass density
constexpr real E = 1e6; // [M/LT^2] Young's modulus
constexpr real nu = 0.3; // [-] Poisson's ratio
constexpr real en = 0.5; // [-] coefficient of normal restitution
constexpr real mus = 0.3; // [-] Coulomb friction coefficient for sliding
constexpr real mur = 0.3; // [-] Coulomb friction coefficient for rolling
constexpr real mut = 3 * PI / 16 * mus; // [-] Coulomb friction coefficient for twisting
constexpr real eta = 1e-2; // [M/LT] dynamic viscosity for drag

constexpr Point v0 = {0,0,-1}; // [L/T] initial linear velocity
constexpr Point w0 = {0,0,0}; // [1/T] initial angular velocity
constexpr Point g = {0,0,-9.81}; // [L/T^2] external acceleration (gravity etc.)

const real dt = Rmin * std::sqrt(rho * (1 - nu*nu) / E); // [T] time step
constexpr real rate = 1e6; // [1/T] particle spawn rate
constexpr std::size_t Nmax = 1000; // [-] maximum number of particles at any time
constexpr std::size_t Nframes = 100; // [-] number of output frames
constexpr std::size_t Nsteps = 500; // [-] number of time steps between frames

// spring-dashpot resistance with Euler update of the spring and Coulomb clipping
template <class T>
T friction(T& u, const real k, const real c, const T& v, const real fmax)
{
  T f = -k * u - c * v; // spring-dashpot resistance
  u = u + dt * v; // forward Euler step for the static friction spring
  const real f2 = f * f; // squared magnitude (dot product for vectors)
  if (f2 > fmax * fmax) // Coulomb's law: dynamic friction?
  {
    f = fmax / std::sqrt(f2) * f; // clip the force
    u = -1 / k * (f + c * v); // set the displacement to match the resistance
  }
  return f; // return the resulting force
}

// add linear and angular accelerations from collisions
template <bool with_mesh>
void collision(Particle& pi, Particle& pj, const std::size_t j)
{
  const Point x = pi.x - pj.x; // distance vector from center to center
  const real x2 = x * x; // squared distance
  const real Ri = pi.R, Rj = pj.R;
  const real Rsum = Ri + Rj;
  if (x2 < Rsum * Rsum) // do the objects overlap?
  {
    const real d = std::sqrt(x2); // distance from center to center
    const Point n = 1 / d * x; // normal vector
    const real depth = Rsum - d; // depth of overlap

    // radii, masses, moments of inertia
    const auto mass    = [](const real R) { return 4 * PI / 3 * rho * R * R * R; }; // mass of a sphere
    const auto inertia = [](const real R, const real m) { return 2 * m * R * R / 5; }; // moment of inertia of a sphere
    const real ri = Ri - depth / 2; // radius of object i to contact point
    const real rj = Rj - depth / 2; // radius of object j to contact point
    const real mi = mass(Ri);
    const real mj = mass(Rj);
    const real Ii = inertia(Ri, mi);
    const real Ij = inertia(Rj, mj);

    // series combination of radii and masses
    const auto eff = [](const real a, const real b) { return with_mesh ? a : a * b / (a + b); };
    const real R = eff(Ri, Rj); // effective radius
    const real r = eff(ri, rj); // effective radius to contact point
    const real m = eff(mi, mj); // effective mass
    const real I = eff(Ii, Ij); // effective moment of inertia

    // stiffness and damping coefficients
    const real a  = std::sqrt(R * depth); // contact radius
    const real kn = 2 * E / (3 * (1 - nu*nu)) * (1 + with_mesh) * a; // normal spring coefficient
    const real ks = 2 * E / ((2 - nu) * (1 + nu)) * (1 + with_mesh) * a; // shearing spring coefficient
    const real kr = kn; // rolling spring coefficient
    const real kt = 2 * (2 - nu) / 3 * ks; // twisting spring coefficient
    const real cn = std::sqrt(5 * m * kn / (1 + std::pow(PI / std::log(en), 2))); // normal damping coefficient
    const real cs = std::sqrt((1 - nu) / (10 * (2 - nu))) * cn; // shearing damping coefficient
    const real cr = cn; // rolling damping coefficient
    const real ct = 2 * std::sqrt(kt * I) / a; // twisting damping coefficient

    // relative linear and angular velocity
    const Point v  = pi.v - pj.v + n.cross(ri * pi.w + rj * pj.w); // relative velocity at contact point
    const Point vn = (v * n) * n; // normal velocity
    const Point vs = v - vn; // sliding velocity
    const Point w  = pi.w - pj.w; // relative angular velocity
    const Point vr = r * w.cross(n); // rolling velocity
    const real  vt = a * (w * n); // twisting velocity

    // determine the contact data to use
    const auto find_or_add = [&]() -> Contact*
    {
      const auto it = std::find_if(pi.c.begin(), pi.c.end(), [j](const Contact& c){ return c.j == j; });
      return &*(it == pi.c.end() ? pi.c.insert(it, Contact{j, {0,0,0}, {0,0,0}, 0, true}) : it);
    };
    Contact* c = nullptr;
    if (with_mesh) // several threads may handle different mesh faces touching the same particle
    {
      #pragma omp critical
      c = find_or_add();
    }
    else // particle-particle: thread i is the only one touching p[i].c
      c = find_or_add();
    c->active = true; // flag it as active so it won't be removed

    // project sliding and rolling friction springs onto current tangent plane
    const auto project = [&n](Point& u)
    {
      const real l = u.length(); // length before
      u = u - (u * n) * n; // subtract normal component
      const real l2 = u.length(); // length after
      if (l2 > 0) u = l / l2 * u; // rescale displacement to previous length
    };
    project(c->us);
    project(c->ur);

    // forces
    const Point fn = kn * depth * n - cn * vn; // normal force vector
    const real  fn_norm = fn.length(); // normal force magnitude
    const Point fs = friction(c->us, ks, cs, vs, mus * fn_norm); // sliding resistance
    const Point fr = friction(c->ur, kr, cr, vr, mur * fn_norm); // rolling resistance
    const real  ft = friction(c->ut, kt, ct, vt, mut * fn_norm); // twisting resistance

    // torques
    const Point taus = ri * fs.cross(n); // sliding
    const Point taur = r * n.cross(fr);  // rolling
    const Point taut = ft * a * n;       // twisting

    // apply the forces and torques
    pi.a.add(1 / mi, fn + fs);
    pi.alpha.add(1 / Ii, taus + taur + taut);
    if (!with_mesh)
    {
      pj.a.add(-1 / mj, fn + fs);
      pj.alpha.add(-1 / Ij, -(rj/ri) * taus + taur + taut);
    }
  }
}

// make a step forward in time
void Ensemble::step()
{
  // compute global bounding box
  Box box;
  #pragma omp parallel
  {
    Box b; // thread-local box
    #pragma omp for nowait
    for (std::size_t i = 0; i < p.size(); ++i)
      b.include(p[i].x, p[i].x);
    #pragma omp critical
    box.include(b.min, b.max);
  }

  // include source box if it is still used
  if (rate > 0 && p.size() < Nmax) // are new particles still being spawned?
    box.include(source.min, source.max);
  else if (p.empty())
    box.include({0,0,0}, {0,0,0}); // allows simulation to run without particles

  // build linked cell list
  const auto cell = [&](const Point& x) { return Cell(x - box.min, 2 * Rmax); }; // cell containing a point
  Nc = cell(box.max); // number of cells in each direction
  ++Nc.x; ++Nc.y; ++Nc.z; // add one cell in all 3 directions because the cell number was rounded down
  head.assign(static_cast<std::size_t>(Nc.x * Nc.y * Nc.z), invalid); // head of particle list in each cell
  for (std::size_t i = 0; i < p.size(); ++i)
  {
    std::size_t& f = first(cell(p[i].x)); // first particle in cell of particle i
    p[i].next = f; // point from particle i to previously first particle in cell
    f = i; // make particle i the first in its cell
  }

  // particle source
  for (real i = 0; i < rate * dt && p.size() < Nmax; ++i)
  {
    const real Rnew = Rmin + rn * (Rmax - Rmin); // random radius
    const auto U = [&](const real lo, const real hi) { return lo + uni_dist(rng) * (hi - lo); }; // uniform in [lo,hi)
    const Point xnew = {U(source.min.x, source.max.x), U(source.min.y, source.max.y), U(source.min.z, source.max.z)};
    const auto hit = [&](const Point& x, const real R) { return (xnew - x) * (xnew - x) < (Rnew + R) * (Rnew + R); };
    bool overlap = false;

    // check distance to other particles in local Moore neighborhood
    const Cell ci = cell(xnew); // cell of the new particle
    for_nearby(ci, ci, [&](const std::size_t j) { overlap = overlap || hit(p[j].x, p[j].R); });

    // check distance to mesh elements (skipped if an overlap was already found)
    for (std::size_t j = 0; j < mesh.f.size() && !overlap; ++j)
      overlap = hit(mesh.closest_point(xnew, j), mesh.f[j].R);

    if (!overlap) // no overlap found with existing particles or mesh elements?
    {
      p.push_back({Rnew, xnew, v0, {0,0,0}, w0, {0,0,0}, {1, {0,0,0}}, first(ci), {}}); // create new particle
      first(ci) = p.size() - 1; // make new particle the first in its cell
      rn = uni_dist(rng); // draw new random number for next particle to spawn
    }
  }

  // compute all accelerations other than from collisions
  #pragma omp parallel for
  for (std::size_t i = 0; i < p.size(); ++i)
  {
    const real c = 9 * eta / (2 * rho * p[i].R * p[i].R); // linear Stokes drag rate
    const real gam = 10 * c / 3; // angular Stokes drag rate
    p[i].a = g - c * p[i].v; // external acceleration (gravity etc.) and linear viscous damping
    p[i].alpha = -gam * p[i].w; // angular viscous damping
  }

  // particle-mesh collisions
  #pragma omp parallel for schedule(dynamic)
  for (std::size_t j = 0; j < mesh.f.size(); ++j)
  {
    Particle pm{}; // mesh element as a static pseudo-particle (zero velocities)
    pm.R = mesh.f[j].R;
    const Cell cmin = cell(mesh.f[j].box.min); // cell of lower box corner
    const Cell cmax = cell(mesh.f[j].box.max); // cell of upper box corner
    for_nearby(cmin, cmax, [&](const std::size_t i)
    {
      pm.x = mesh.closest_point(p[i].x, j);
      collision<true>(p[i], pm, invalid - j);
    });
  }

  // particle-particle collisions
  #pragma omp parallel for
  for (std::size_t i = 0; i < p.size(); ++i)
  {
    const Cell ci = cell(p[i].x); // cell of particle i
    for_nearby(ci, ci, [&](const std::size_t j) { if (i < j) collision<false>(p[i], p[j], j); });
    p[i].c.remove_if([](Contact& c) { return c.active = !c.active; }); // remove history of inactive contacts
  }

  // time integration (semi-implicit Euler method for translation, SPIRAL for rotation)
  #pragma omp parallel for
  for (std::size_t i = 0; i < p.size(); ++i)
  {
    p[i].v = p[i].v + dt * p[i].a; // new velocity
    p[i].x = p[i].x + dt * p[i].v; // new position
    p[i].q.update(dt, p[i].w, p[i].alpha); // update quaternion before angular velocity
    p[i].w = p[i].w + dt * p[i].alpha; // new angular velocity
  }
  t += dt; // advance the time

  // particle sink
  if (sink.contains(sink.min)) // non-empty sink region?
  {
    std::size_t n = 0; // number of remaining particles
    for (Particle& P : p)
      P.next = sink.contains(P.x) ? invalid : n++; // new index (reusing Particle::next as a temporary)
    #pragma omp parallel for
    for (std::size_t i = 0; i < p.size(); ++i)
      for (Contact& c : p[i].c)
        if (c.j < p.size()) // skip mesh faces
          c.j = p[c.j].next; // renumber contacts
    p.erase(std::remove_if(p.begin(), p.end(), [](const Particle& P) { return P.next == invalid; }), p.end());
  }
}

int main(int argc, char** argv)
{
  const std::string mfile  = (argc > 1 ? argv[1] : ""); // optional mesh file (Object File Format)
  const std::string pfile  = (argc > 2 ? argv[2] : ""); // optional particle input file (CSV file in same format as output)
  const std::string outdir = (argc > 3 ? argv[3] : "output"); // output directory

  const Mesh mesh(mfile, Rmesh); // generate mesh
  Ensemble ensemble(pfile, mesh); // generate particle ensemble

  if (std::system((std::string("mkdir -p '") + outdir + "'").c_str())) return 1; // create output directory, abort if failed

  ensemble.output(outdir, 0); // print initial state
  for (std::size_t f = 1; f <= Nframes; ++f)
  {
    for (std::size_t s = 0; s < Nsteps; ++s)
      ensemble.step(); // make a step forward in time
    ensemble.output(outdir, f); // print a frame
  }
}
