def opt2file(opt,des_file):
   args = vars(opt)
   with open(des_file,'wt') as opt_file:
       opt_file.write('-----------options-----------\n')
       print('----------options----------\n')
       for k,v in sorted(args.items()):
           opt_file.write('%s: %s\n' %(str(k),str(v)))
           print("%s: %s\n" %(str(k),str(v)))
   opt_file('---------End-----------')
   print("---------End-----------")


