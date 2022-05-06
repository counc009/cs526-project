#define CHECK2(type,field1,field2,unique) \
struct check2_ ## unique {\
  double avg;\
  int paths;\
  int paths_field1;\
  double count_field1;\
  int paths_field2;\
  double count_field2;\
};\
\
struct check2_ ## unique Check2Routine_ ## unique (type p) {\
  struct check2_ ## unique f1result,f2result;\
  struct check2_ ## unique  myresult;\
  type tmp;\
  \
  MIGRPH();\
  if (p==NULL) {\
     myresult.avg = 0.0;\
     myresult.paths = 0;\
     myresult.paths_field1 = myresult.paths_field2 = 0;\
     myresult.count_field1 = myresult.count_field2 = 0.0;\
     return myresult;\
  }\
  if (p->field1==NULL && p->field2==NULL) {\
    myresult.avg = 1.0;\
    myresult.paths = 1;\
    myresult.paths_field1 = myresult.paths_field2 = 0;\
    myresult.count_field1 = myresult.count_field2 = 0;\
    return myresult;\
  }\
  tmp = p->field1;\
  f1result = Check2Routine_ ## unique (tmp);\
  tmp = p->field2;\
  f2result = Check2Routine_ ## unique (tmp);\
  myresult.avg = 0.0;\
  myresult.paths = f1result.paths+f2result.paths;\
  myresult.paths_field1 = f1result.paths_field1 + f2result.paths_field1; \
  myresult.paths_field2 = f1result.paths_field2 + f2result.paths_field2; \
  myresult.count_field1 = f1result.count_field1 + f2result.count_field1; \
  myresult.count_field2 = f1result.count_field2 + f2result.count_field2; \
  if (ISLOCPTR(p->field1)) {\
    myresult.avg += (f1result.avg)*((double) f1result.paths);\
  }\
  else {\
    myresult.paths_field1 +=f1result.paths;\
    myresult.count_field1 +=(((double) f1result.paths)*f1result.avg);\
  }\
  if (ISLOCPTR(p->field2)) {\
    myresult.avg += (f2result.avg)*((double) f2result.paths);\
  }\
  else {\
    myresult.paths_field2 +=f2result.paths;\
    myresult.count_field2 +=(((double) f2result.paths)*f2result.avg);\
  }\
  myresult.avg /= myresult.paths;\
  myresult.avg += 1.0;\
  UNPHASE();\
  return myresult;\
}  \
\
void Docheck_ ## unique (type p) { \
  struct check2_ ## unique x; \
  int affinity1,affinity2; \
\
  MIGRPH();\
  x = Check2Routine_ ## unique(p);\
  if (!ISLOCPTR(p)) {\
    x.paths_field2 +=x.paths;\
    x.count_field2 +=(((double) x.paths)*x.avg);\
    x.paths_field1 +=x.paths;\
    x.count_field1 +=(((double) x.paths)*x.avg);\
  }\
  affinity1 = (x.paths_field1) ? 100.0*(1.0 - 1.0/(x.count_field1 / (double) x.paths_field1)) : 100;\
  affinity2 = (x.paths_field2) ? 100.0*(1.0 - 1.0/(x.count_field2 / (double) x.paths_field2)) : 100;\
  printf("Affinity of field " # field1 " is %d\n",affinity1); \
  printf("Affinity of field " # field2 " is %d\n",affinity2); \
  UNPHASE();\
}
  
#define CHECK1(type,field1,unique) \
struct check1_ ## unique {\
  double avg;\
  int paths;\
  int paths_field1;\
  double count_field1;\
};\
\
struct check1_ ## unique Check1Routine_ ## unique (type p, type first) {\
  struct check1_ ## unique f1result;\
  struct check1_ ## unique myresult;\
  type tmp;\
  \
  MIGRPH();\
  if (p==NULL) {\
     myresult.avg = 0.0;\
     myresult.paths = 0;\
     myresult.paths_field1 = 0;\
     myresult.count_field1 = 0.0;\
     return myresult;\
  }\
  if (p->field1==NULL || p->field1==first) {\
    myresult.avg = 1.0;\
    myresult.paths = 1;\
    myresult.paths_field1 = 0;\
    myresult.count_field1 = 0.0;\
    return myresult;\
  }\
  tmp = p->field1;\
  f1result = Check1Routine_ ## unique (tmp,first);\
  myresult.avg = 0.0;\
  myresult.paths = f1result.paths;\
  myresult.paths_field1 = f1result.paths_field1;\
  myresult.count_field1 = f1result.count_field1;\
  if (ISLOCPTR(p->field1)) {\
    myresult.avg += (f1result.avg)*((double) f1result.paths);\
  }\
  else {\
    myresult.paths_field1 +=f1result.paths;\
    myresult.count_field1 +=(((double) f1result.paths)*f1result.avg);\
  }\
  myresult.avg /= myresult.paths;\
  myresult.avg += 1.0;\
  UNPHASE();\
  return myresult;\
}  \
\
void Docheck_ ## unique (type p) { \
  struct check1_ ## unique x; \
  int affinity1,affinity1; \
\
  MIGRPH();\
  x = Check1Routine_ ## unique(p,p);\
  if (!ISLOCPTR(p)) {\
    x.paths_field1 +=x.paths;\
    x.count_field1 +=(((double) x.paths)*x.avg);\
  }\
  affinity1 = (x.paths_field1) ? 100.0*(1.0 - 1.0/(x.count_field1 / (double) x.paths_field1)) : 100;\
  printf("Affinity of field " # field1 " is %d\n",affinity1); \
  UNPHASE();\
}
  
#define CHECK4(type,field1,field2,field3,field4,unique) \
struct check4_ ## unique {\
  double avg;\
  int paths;\
  int paths_field1;\
  double count_field1;\
  int paths_field2;\
  double count_field2;\
  int paths_field3;\
  double count_field3;\
  int paths_field4;\
  double count_field4;\
};\
\
struct check4_ ## unique Check4Routine_ ## unique (type p) {\
  struct check4_ ## unique f1result,f2result,f3result,f4result;\
  struct check4_ ## unique  myresult;\
  type tmp;\
  \
  MIGRPH();\
  if (p==NULL) {\
     myresult.avg = 0.0;\
     myresult.paths = 0;\
     myresult.paths_field1 = myresult.paths_field2 = 0;\
     myresult.count_field1 = myresult.count_field2 = 0.0;\
     myresult.paths_field3 = myresult.paths_field4 = 0;\
     myresult.count_field3 = myresult.count_field4 = 0.0;\
     return myresult;\
  }\
  if (p->field1==NULL && p->field2==NULL && p->field3==NULL && p->field4==NULL) {\
    myresult.avg = 1.0;\
    myresult.paths = 1;\
    myresult.paths_field1 = myresult.paths_field2 = 0;\
    myresult.count_field1 = myresult.count_field2 = 0.0;\
    myresult.paths_field3 = myresult.paths_field4 = 0;\
    myresult.count_field3 = myresult.count_field4 = 0.0;\
    return myresult;\
  }\
  tmp = p->field1;\
  f1result = Check4Routine_ ## unique (tmp);\
  tmp = p->field2;\
  f2result = Check4Routine_ ## unique (tmp);\
  tmp = p->field3;\
  f3result = Check4Routine_ ## unique (tmp);\
  tmp = p->field4;\
  f4result = Check4Routine_ ## unique (tmp);\
  myresult.avg = 0.0;\
  myresult.paths = f1result.paths+f2result.paths+f3result.paths+f4result.paths;\
  myresult.paths_field1 = f1result.paths_field1 + f2result.paths_field1 + f3result.paths_field1 + f4result.paths_field1; \
  myresult.paths_field2 = f1result.paths_field2 + f2result.paths_field2 + f3result.paths_field2 + f4result.paths_field2; \
  myresult.count_field1 = f1result.count_field1 + f2result.count_field1 + f3result.count_field1 + f4result.count_field1; \
  myresult.count_field2 = f1result.count_field2 + f2result.count_field2 + f3result.count_field2 + f4result.count_field2; \
  myresult.paths_field3 = f1result.paths_field3 + f2result.paths_field3 + f3result.paths_field3 + f4result.paths_field3; \
  myresult.paths_field4 = f1result.paths_field4 + f2result.paths_field4 + f3result.paths_field4 + f4result.paths_field4; \
  myresult.count_field3 = f1result.count_field3 + f2result.count_field3 + f3result.count_field3 + f4result.count_field3; \
  myresult.count_field4 = f1result.count_field4 + f2result.count_field4 + f3result.count_field4 + f4result.count_field4; \
  if (ISLOCPTR(p->field1)) {\
    myresult.avg += (f1result.avg)*((double) f1result.paths);\
  }\
  else {\
    myresult.paths_field1 +=f1result.paths;\
    myresult.count_field1 +=(((double) f1result.paths)*f1result.avg);\
  }\
  if (ISLOCPTR(p->field2)) {\
    myresult.avg += (f2result.avg)*((double) f2result.paths);\
  }\
  else {\
    myresult.paths_field2 +=f2result.paths;\
    myresult.count_field2 +=(((double) f2result.paths)*f2result.avg);\
  }\
  if (ISLOCPTR(p->field3)) {\
    myresult.avg += (f3result.avg)*((double) f3result.paths);\
  }\
  else {\
    myresult.paths_field3 +=f3result.paths;\
    myresult.count_field3 +=(((double) f3result.paths)*f3result.avg);\
  }\
  if (ISLOCPTR(p->field4)) {\
    myresult.avg += (f4result.avg)*((double) f4result.paths);\
  }\
  else {\
    myresult.paths_field4 +=f4result.paths;\
    myresult.count_field4 +=(((double) f4result.paths)*f4result.avg);\
  }\
  myresult.avg /= myresult.paths;\
  myresult.avg += 1.0;\
  UNPHASE();\
  return myresult;\
}  \
\
void Docheck_ ## unique (type p) { \
  struct check4_ ## unique x; \
  int affinity1,affinity2,affinity3,affinity4; \
\
  MIGRPH();\
  x = Check4Routine_ ## unique(p);\
  if (!ISLOCPTR(p)) {\
    x.paths_field4 +=x.paths;\
    x.count_field4 +=(((double) x.paths)*x.avg);\
    x.paths_field3 +=x.paths;\
    x.count_field3 +=(((double) x.paths)*x.avg);\
    x.paths_field2 +=x.paths;\
    x.count_field2 +=(((double) x.paths)*x.avg);\
    x.paths_field1 +=x.paths;\
    x.count_field1 +=(((double) x.paths)*x.avg);\
  }\
  affinity1 = (x.paths_field1) ? 100.0*(1.0 - 1.0/(x.count_field1 / (double) x.paths_field1)) : 100;\
  affinity2 = (x.paths_field2) ? 100.0*(1.0 - 1.0/(x.count_field2 / (double) x.paths_field2)) : 100;\
  affinity3 = (x.paths_field3) ? 100.0*(1.0 - 1.0/(x.count_field3 / (double) x.paths_field3)) : 100;\
  affinity4 = (x.paths_field4) ? 100.0*(1.0 - 1.0/(x.count_field4 / (double) x.paths_field4)) : 100;\
  printf("Affinity of field " # field1 " is %d\n",affinity1); \
  printf("Affinity of field " # field2 " is %d\n",affinity2); \
  printf("Affinity of field " # field3 " is %d\n",affinity3); \
  printf("Affinity of field " # field4 " is %d\n",affinity4); \
  UNPHASE();\
}
  
#define CHECK8(type,isleaf,field1,field2,field3,field4,field5,field6,field7,field8,unique) \
struct check8_ ## unique {\
  double avg;\
  int paths;\
  int paths_field1;\
  double count_field1; \
  int paths_field2;\
  double count_field2; \
  int paths_field3;\
  double count_field3; \
  int paths_field4;\
  double count_field4; \
  int paths_field5;\
  double count_field5; \
  int paths_field6;\
  double count_field6; \
  int paths_field7;\
  double count_field7; \
  int paths_field8;\
  double count_field8; \
};\
\
struct check8_ ## unique Check8Routine_ ## unique (type p) {\
  struct check8_ ## unique f1result,f2result,f3result,f4result;\
  struct check8_ ## unique f5result,f6result,f7result,f8result;\
  struct check8_ ## unique  myresult;\
  type tmp;\
  \
  MIGRPH();\
  if (p==NULL) {\
     myresult.avg = 0.0;\
     myresult.paths = 0;\
     myresult.paths_field1 = myresult.paths_field2 = 0;\
     myresult.count_field1 = myresult.count_field2 = 0.0;\
     myresult.paths_field3 = myresult.paths_field4 = 0;\
     myresult.count_field3 = myresult.count_field4 = 0.0;\
     myresult.paths_field5 = myresult.paths_field6 = 0;\
     myresult.count_field5 = myresult.count_field6 = 0.0;\
     myresult.paths_field7 = myresult.paths_field8 = 0;\
     myresult.count_field7 = myresult.count_field8 = 0.0;\
     return myresult;\
  }\
 if (isleaf) {\
    myresult.avg = 1.0;\
    myresult.paths = 1;\
    myresult.paths_field1 = myresult.paths_field2 = 0;\
    myresult.count_field1 = myresult.count_field2 = 0.0;\
    myresult.paths_field3 = myresult.paths_field4 = 0;\
    myresult.count_field3 = myresult.count_field4 = 0.0;\
    myresult.paths_field5 = myresult.paths_field6 = 0;\
    myresult.count_field5 = myresult.count_field6 = 0.0;\
    myresult.paths_field7 = myresult.paths_field8 = 0;\
    myresult.count_field7 = myresult.count_field8 = 0.0;\
    return myresult;\
  }\
  tmp = (type) p->field1;\
  f1result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field2;\
  f2result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field3;\
  f3result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field4;\
  f4result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field5;\
  f5result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field6;\
  f6result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field7;\
  f7result = Check8Routine_ ## unique (tmp);\
  tmp = (type) p->field8;\
  f8result = Check8Routine_ ## unique (tmp);\
  myresult.avg = 0.0;\
  myresult.paths = f1result.paths+f2result.paths+f3result.paths+f4result.paths;\
  myresult.paths +=f5result.paths+f6result.paths+f7result.paths+f8result.paths;\
  myresult.paths_field1 = f1result.paths_field1 + f2result.paths_field1 + f3result.paths_field1 + f4result.paths_field1; \
  myresult.paths_field2 = f1result.paths_field2 + f2result.paths_field2 + f3result.paths_field2 + f4result.paths_field2; \
  myresult.count_field1 = f1result.count_field1 + f2result.count_field1 + f3result.count_field1 + f4result.count_field1; \
  myresult.count_field2 = f1result.count_field2 + f2result.count_field2 + f3result.count_field2 + f4result.count_field2; \
  myresult.paths_field3 = f1result.paths_field3 + f2result.paths_field3 + f3result.paths_field3 + f4result.paths_field3; \
  myresult.paths_field4 = f1result.paths_field4 + f2result.paths_field4 + f3result.paths_field4 + f4result.paths_field4; \
  myresult.count_field3 = f1result.count_field3 + f2result.count_field3 + f3result.count_field3 + f4result.count_field3; \
  myresult.count_field4 = f1result.count_field4 + f2result.count_field4 + f3result.count_field4 + f4result.count_field4; \
  myresult.paths_field5 = f1result.paths_field5 + f2result.paths_field5 + f3result.paths_field5 + f4result.paths_field5; \
  myresult.paths_field6 = f1result.paths_field6 + f2result.paths_field6 + f3result.paths_field6 + f4result.paths_field6; \
  myresult.count_field5 = f1result.count_field5 + f2result.count_field5 + f3result.count_field5 + f4result.count_field5; \
  myresult.count_field6 = f1result.count_field6 + f2result.count_field6 + f3result.count_field6 + f4result.count_field6; \
  myresult.paths_field7 = f1result.paths_field7 + f2result.paths_field7 + f3result.paths_field7 + f4result.paths_field7; \
  myresult.paths_field8 = f1result.paths_field8 + f2result.paths_field8 + f3result.paths_field8 + f4result.paths_field8; \
  myresult.count_field7 = f1result.count_field7 + f2result.count_field7 + f3result.count_field7 + f4result.count_field7; \
  myresult.count_field8 = f1result.count_field8 + f2result.count_field8 + f3result.count_field8 + f4result.count_field8; \
  myresult.paths_field1 += f5result.paths_field1 + f6result.paths_field1 + f7result.paths_field1 + f8result.paths_field1; \
  myresult.paths_field2 += f5result.paths_field2 + f6result.paths_field2 + f7result.paths_field2 + f8result.paths_field2; \
  myresult.count_field1 += f5result.count_field1 + f6result.count_field1 + f7result.count_field1 + f8result.count_field1; \
  myresult.count_field2 += f5result.count_field2 + f6result.count_field2 + f7result.count_field2 + f8result.count_field2; \
  myresult.paths_field3 += f5result.paths_field3 + f6result.paths_field3 + f7result.paths_field3 + f8result.paths_field3; \
  myresult.paths_field4 += f5result.paths_field4 + f6result.paths_field4 + f7result.paths_field4 + f8result.paths_field4; \
  myresult.count_field3 += f5result.count_field3 + f6result.count_field3 + f7result.count_field3 + f8result.count_field3; \
  myresult.count_field4 += f5result.count_field4 + f6result.count_field4 + f7result.count_field4 + f8result.count_field4; \
  myresult.paths_field5 += f5result.paths_field5 + f6result.paths_field5 + f7result.paths_field5 + f8result.paths_field5; \
  myresult.paths_field6 += f5result.paths_field6 + f6result.paths_field6 + f7result.paths_field6 + f8result.paths_field6; \
  myresult.count_field5 += f5result.count_field5 + f6result.count_field5 + f7result.count_field5 + f8result.count_field5; \
  myresult.count_field6 += f5result.count_field6 + f6result.count_field6 + f7result.count_field6 + f8result.count_field6; \
  myresult.paths_field7 += f5result.paths_field7 + f6result.paths_field7 + f7result.paths_field7 + f8result.paths_field7; \
  myresult.paths_field8 += f5result.paths_field8 + f6result.paths_field8 + f7result.paths_field8 + f8result.paths_field8; \
  myresult.count_field7 += f5result.count_field7 + f6result.count_field7 + f7result.count_field7 + f8result.count_field7; \
  myresult.count_field8 += f5result.count_field8 + f6result.count_field8 + f7result.count_field8 + f8result.count_field8; \
  if (ISLOCPTR(p->field1)) {\
    myresult.avg += (f1result.avg)*((double) f1result.paths);\
  }\
  else {\
    myresult.paths_field1 +=f1result.paths;\
    myresult.count_field1 +=(((double) f1result.paths)*f1result.avg);\
  }\
  if (ISLOCPTR(p->field2)) {\
    myresult.avg += (f2result.avg)*((double) f2result.paths);\
  }\
  else {\
    myresult.paths_field2 +=f2result.paths;\
    myresult.count_field2 +=(((double) f2result.paths)*f2result.avg);\
  }\
  if (ISLOCPTR(p->field3)) {\
    myresult.avg += (f3result.avg)*((double) f3result.paths);\
  }\
  else {\
    myresult.paths_field3 +=f3result.paths;\
    myresult.count_field3 +=(((double) f3result.paths)*f3result.avg);\
  }\
  if (ISLOCPTR(p->field4)) {\
    myresult.avg += (f4result.avg)*((double) f4result.paths);\
  }\
  else {\
    myresult.paths_field4 +=f4result.paths;\
    myresult.count_field4 +=(((double) f4result.paths)*f4result.avg);\
  }\
  if (ISLOCPTR(p->field5)) {\
    myresult.avg += (f5result.avg)*((double) f5result.paths);\
  }\
  else {\
    myresult.paths_field5 +=f5result.paths;\
    myresult.count_field5 +=(((double) f5result.paths)*f5result.avg);\
  }\
  if (ISLOCPTR(p->field6)) {\
    myresult.avg += (f6result.avg)*((double) f6result.paths);\
  }\
  else {\
    myresult.paths_field6 +=f6result.paths;\
    myresult.count_field6 +=(((double) f6result.paths)*f6result.avg);\
  }\
  if (ISLOCPTR(p->field7)) {\
    myresult.avg += (f7result.avg)*((double) f7result.paths);\
  }\
  else {\
    myresult.paths_field7 +=f7result.paths;\
    myresult.count_field7 +=(((double) f7result.paths)*f7result.avg);\
  }\
  if (ISLOCPTR(p->field8)) {\
    myresult.avg += (f8result.avg)*((double) f8result.paths);\
  }\
  else {\
    myresult.paths_field8 +=f8result.paths;\
    myresult.count_field8 +=(((double) f8result.paths)*f8result.avg);\
  }\
  myresult.avg /= myresult.paths;\
  myresult.avg += 1.0;\
  UNPHASE();\
  return myresult;\
}  \
\
void Docheck_ ## unique (type p) { \
  struct check8_ ## unique x; \
  int affinity1,affinity2,affinity3,affinity4; \
  int affinity5,affinity6,affinity7,affinity8; \
\
  MIGRPH();\
  x = Check8Routine_ ## unique(p);\
  if (!ISLOCPTR(p)) {\
    x.paths_field8 +=x.paths;\
    x.count_field8 +=(((double) x.paths)*x.avg);\
    x.paths_field7 +=x.paths;\
    x.count_field7 +=(((double) x.paths)*x.avg);\
    x.paths_field6 +=x.paths;\
    x.count_field6 +=(((double) x.paths)*x.avg);\
    x.paths_field5 +=x.paths;\
    x.count_field5 +=(((double) x.paths)*x.avg);\
    x.paths_field4 +=x.paths;\
    x.count_field4 +=(((double) x.paths)*x.avg);\
    x.paths_field3 +=x.paths;\
    x.count_field3 +=(((double) x.paths)*x.avg);\
    x.paths_field2 +=x.paths;\
    x.count_field2 +=(((double) x.paths)*x.avg);\
    x.paths_field1 +=x.paths;\
    x.count_field1 +=(((double) x.paths)*x.avg);\
  }\
  affinity1 = (x.paths_field1) ? 100.0*(1.0 - 1.0/(x.count_field1 / (double) x.paths_field1)) : 100;\
  affinity2 = (x.paths_field2) ? 100.0*(1.0 - 1.0/(x.count_field2 / (double) x.paths_field2)) : 100;\
  affinity3 = (x.paths_field3) ? 100.0*(1.0 - 1.0/(x.count_field3 / (double) x.paths_field3)) : 100;\
  affinity4 = (x.paths_field4) ? 100.0*(1.0 - 1.0/(x.count_field4 / (double) x.paths_field4)) : 100;\
  affinity5 = (x.paths_field5) ? 100.0*(1.0 - 1.0/(x.count_field5 / (double) x.paths_field5)) : 100;\
  affinity6 = (x.paths_field6) ? 100.0*(1.0 - 1.0/(x.count_field6 / (double) x.paths_field6)) : 100;\
  affinity7 = (x.paths_field7) ? 100.0*(1.0 - 1.0/(x.count_field7 / (double) x.paths_field7)) : 100;\
  affinity8 = (x.paths_field8) ? 100.0*(1.0 - 1.0/(x.count_field8 / (double) x.paths_field8)) : 100;\
  printf("Affinity of field " # field1 " is %d\n",affinity1); \
  printf("Affinity of field " # field2 " is %d\n",affinity2); \
  printf("Affinity of field " # field3 " is %d\n",affinity3); \
  printf("Affinity of field " # field4 " is %d\n",affinity4); \
  printf("Affinity of field " # field5 " is %d\n",affinity5); \
  printf("Affinity of field " # field6 " is %d\n",affinity6); \
  printf("Affinity of field " # field7 " is %d\n",affinity7); \
  printf("Affinity of field " # field8 " is %d\n",affinity8); \
  UNPHASE();\
}
  
#define CHECK1_ACCUM(type,field1,unique) \
int Global_total_ ## unique ## _ ## field1 = 0; \
int Global_nonloc_ ## unique ## _ ## field1 = 0; \
\
void Accumulate_ ## unique (type p, type first) \
{\
  MIGRPH();\
  if (NONLOCAL(p)) Global_nonloc_ ## unique ## _ ## field1 ++;\
  Global_total_ ## unique ## _ ## field1 ++;\
  while (p->field1 && (p->field1 != first)) {\
    Global_total_ ## unique ## _ ## field1 ++;\
    if (NONLOCAL(p->field1)) Global_nonloc_ ## unique ## _ ## field1 ++;\
    p = p->field1;\
  }\
  UNPHASE();\
}\
\
void Print_Accumulated_ ## unique () \
{\
  int i, affinity;\
  int accum_nonloc_ ## field1 = 0;\
  int accum_total_ ## field1 = 0;\
  for (i=0; i<__NumNodes; i++) {\
    MIGRATE(i);\
    accum_nonloc_ ## field1 += Global_nonloc_ ## unique ## _ ## field1 ;\
    accum_total_ ## field1 += Global_total_ ## unique ## _ ## field1 ;\
  }\
  if (accum_total_ ## field1) \
    affinity = 100 - (100 * accum_nonloc_ ## field1) / \
               accum_total_ ## field1;\
  else affinity = 100;\
  printf("[%d]: Accumulated affinity for " #field1 " is %d\n",__MyNodeId,affinity);\
}
