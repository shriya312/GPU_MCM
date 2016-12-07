import os, sys
import math
fp = open ('f','r')
search1 = str(sys.argv[1])
search2 = str(sys.argv[2])
num = int (sys.argv[3])
time1 = 0
time2 = 0
for line in fp:
	if "TIME" in line:
		l = line.strip()
		l1 = l.split()
		if search1 in line:
			time1 = float (l1[-1]);
		elif search2 in line:
			time2 = float ( l1[-1]);

fp.close()
log_num = int (math.log(num,2));

print "%d %f %f"%(log_num,time1,time2)
