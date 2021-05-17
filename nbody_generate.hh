#ifndef GENERATE_VANILLA_H
#define GENERATE_VANILLA_H

// put particles in a box of size
// masses vary randomly by mdelta around m0
// velocity is zero
// center of mass adjusted to zero
template<typename doubleX>
void cube (int n, long int seed, double size, double m0, double mdelta,  
           doubleX x[], doubleX v[], double m[])
{
  int i;
  double s[3] = {0.0,0.0,0.0};
  double t[3] = {0.0,0.0,0.0};
  double M = 0.0;

  if (seed!=0) srand48(seed);
  for (i=0; i<n; i++)
    {
      x[i][0] = drand48()*size;
      x[i][1] = drand48()*size;
      x[i][2] = drand48()*size;
      v[i][0] = 0.0;
      v[i][1] = 0.0;
      v[i][2] = 0.0;
      m[i] = m0 + (drand48()-0.5)*2.0*mdelta;
      s[0] += m[i]*x[i][0];
      s[1] += m[i]*x[i][1];
      s[2] += m[i]*x[i][2];
      t[0] += m[i]*v[i][0];
      t[1] += m[i]*v[i][1];
      t[2] += m[i]*v[i][2];
      M += m[i];
    }
  printf("center of mass: %g %g %g\n",s[0],s[1],s[2]);
  for (i=0; i<n; i++)
    {
      x[i][0] -= s[0]; 
      x[i][1] -= s[1]; 
      x[i][2] -= s[2]; 

      v[i][0] -= t[0]; 
      v[i][1] -= t[1]; 
      v[i][2] -= t[2]; 
    }
}


/* This is a copy from 
 *  Pit Hut, Jun Makino: The Art of Computational Science, 
 *  The Kali Code, vol. 5. Initial Conditions: Plummer's Model.
 */ 
template<typename doubleX>
double plummer (int n, long int seed,
                doubleX x[], doubleX v[], double m[], const int verbosity=1)
{
  int i;
  double radius,theta,phi,X,Y,velocity,maxr=-1.0;
  const double Pi = 3.141592653589793238462643383279;
  double s[3] = {0.0,0.0,0.0};
  double t[3] = {0.0,0.0,0.0};
  if (seed!=0) srand48(seed);
  for (i=0; i<n; i++)
    {
      m[i] = 1.0/n;
      radius = 1.0/sqrt(pow(drand48(),-2.0/3.0)-1.0);
      if (radius>maxr) maxr=radius;
      theta = acos(-1.0+drand48()*2.0);
      phi = drand48()*2.0*Pi;
      x[i][0] = radius*sin(theta)*cos(phi);
      x[i][1] = radius*sin(theta)*sin(phi);
      x[i][2] = radius*cos(theta);
      s[0] += m[i]*x[i][0];
      s[1] += m[i]*x[i][1];
      s[2] += m[i]*x[i][2];
      X = 0.0;
      Y = 0.1;
      while (Y>X*X*pow(1.0-X*X,3.5))
        {
          X = drand48();
          Y = drand48()*0.1;
        }
      velocity = X*sqrt(2.0)*pow(1.0+radius*radius,-0.25);
      theta = acos(-1.0+drand48()*2.0);
      phi = drand48()*2.0*Pi;
      v[i][0] = velocity*sin(theta)*cos(phi);
      v[i][1] = velocity*sin(theta)*sin(phi);
      v[i][2] = velocity*cos(theta);
      t[0] += m[i]*v[i][0];
      t[1] += m[i]*v[i][1];
      t[2] += m[i]*v[i][2];
    }
  if (verbosity>0)
    printf("center of mass: %g %g %g\n",s[0],s[1],s[2]);
  for (i=0; i<n; i++)
    {
      x[i][0] -= s[0]; 
      x[i][1] -= s[1]; 
      x[i][2] -= s[2]; 

      v[i][0] -= t[0]; 
      v[i][1] -= t[1]; 
      v[i][2] -= t[2]; 
    }
  s[0]=s[1]=s[2]=0.0;
  for (i=0; i<n; i++)
    {
      s[0] += m[i]*x[i][0];
      s[1] += m[i]*x[i][1];
      s[2] += m[i]*x[i][2];
    }
  if (verbosity>0){
    printf("new center of mass: %g %g %g\n",s[0],s[1],s[2]);
    printf("maximum radius: %g\n",maxr);
  }
  return maxr;
}

// put two plummer type clusters t some distance
// and with slight tngential velocity component
template<typename doubleX>
void two_plummer (int n, long int seed,
                  doubleX x[], doubleX v[], double m[], const int verbosity=1)
{
  // check if n iss even
  if (n%2!=0)
    {
      printf("two_plummer needs n to be even");
      exit(1);
    }
  // make a plummer distribution in the first half
  auto radius = plummer(n/2,seed,x,v,m, verbosity);
  // define velocity
  double V = radius/1000.0;
  if (verbosity>0)
    std::cout << "start velocity is " << V << std::endl;
  // copy to second half at a distance of radius
  for (int i=0; i<n/2; i++)
    {
      m[i+n/2] = m[i];
      x[i+n/2][0] = x[i][0]+radius; 
      x[i+n/2][1] = x[i][1]; 
      x[i+n/2][2] = x[i][2]; 
      v[i+n/2][0] = v[i][0]; 
      v[i+n/2][1] = v[i][1]+V; 
      v[i+n/2][2] = v[i][2]; 
    }
  // recenter
  double s[3] = {0.0,0.0,0.0};
  double t[3] = {0.0,0.0,0.0};
  for (int i=0; i<n; i++)
    {
      s[0] += m[i]*x[i][0];
      s[1] += m[i]*x[i][1];
      s[2] += m[i]*x[i][2];
      t[0] += m[i]*v[i][0];
      t[1] += m[i]*v[i][1];
      t[2] += m[i]*v[i][2];
    }
  for (int i=0; i<n; i++)
    {
      x[i][0] -= s[0]; 
      x[i][1] -= s[1]; 
      x[i][2] -= s[2]; 
      // v[i][0] -= t[0]; 
      // v[i][1] -= t[1]; 
      // v[i][2] -= t[2]; 
    }
}

#endif
