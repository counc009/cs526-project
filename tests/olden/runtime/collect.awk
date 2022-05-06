BEGIN { lastfile = "NONE"; lastline = -999; s1 = 0; s2 = 0; s3 = 0; s4 = 0; cm = 0;}

(($2 == lastfile) && ($4 == lastline) && ($7 == "Migrates:")) {
  s1 += $6; s2 += $8; cm = 1; 
}

(($2 == lastfile) && ($4 == lastline) && ($7 == "Hit:")) {
  s1 += $6; s2 += $8; s3 += $10; s4 += $12; cm = 0;
}

(($2 != lastfile) || ($4 != lastline)) {
if (cm == 1) {
  if (lastline != -999) {
    printf("File: %s Line: %05d Local: %d Migrates: %d\n",lastfile,lastline,s1,s2);
  }
  lastfile = $2; lastline = $4;
  s1 = 0; s2 = 0; s3 = 0; s4 = 0;
}

else if (cm == 0) {
  if (lastline != -999) {
    printf("File: %s Line: %05d Loc: %d Hit: %d Miss: %d Rem_Wr: %d\n",lastfile,lastline,s1,s2,s3,s4);
  }
  lastfile = $2; lastline = $4;
  s1 = 0; s2 = 0; s3 = 0; s4 = 0;
}

if ($7 == "Migrates:") {
  s1 = $6; s2 = $8; cm = 1;
}
if ($7 == "Hit:") {
  s1 = $6; s2 = $8; s3 = $10; s4 = $12; cm = 0;
}
}
   
END {
  if (cm == 0) 
    printf("File: %s Line: %05d Loc: %d Hit: %d Miss: %d Rem_Wr: %d\n",lastfile,lastline,s1,s2,s3,s4);
  else
    printf("File: %s Line: %05d Local: %d Migrates: %d\n",lastfile,lastline,s1,s2);
}
