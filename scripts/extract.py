import os, sys

fp = open ('f','r')
gpu_time = 0
cpu_time = 0
for line in fp:
	if "TIME" in line:
		l = line.strip()
		l1 = l.split()
		if "GPU" in line:
			gpu_time = float (l1[-1]);
		elif "SERIAL" in line:
			cpu_time = float ( l1[-1]);

fp.close()

print "%f %f"%(cpu_time,gpu_time)
