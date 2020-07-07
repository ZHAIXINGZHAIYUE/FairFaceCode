import os 
import sys

def main():
    in1 = sys.argv[1]
    in2 = sys.argv[2]
    in3 = sys.argv[3]
    out = sys.argv[4]

    with open(in1,'r') as fin1, open(in2, 'r') as fin2,open(in3, 'r') as fin3, open(out, 'w') as fout:
        for i,(line1, line2, line3) in enumerate(zip(fin1.readlines(), fin2.readlines(), fin3.readlines())):
            if i == 0:
                fout.write(line1)
            else:
                name1,name2,sim1 = line1.strip().split(',')
                _,_,sim2 = line2.strip().split(',')
                _,_,sim3 = line3.strip().split(',')
                sim = float(sim1) + float(sim2) + float(sim3)
                fout.write('{},{},{}\n'.format(name1, name2, sim))

if __name__ == '__main__':
    main()    
