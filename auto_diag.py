# c = creation (above interaction line)
# a = annihilation (below interaction line)
# h = hole (downward arrow)
# p = particle (upward arrow)

# h(c) h(a) ---> col 1
# p(c) p(a) ---> col 2
# h(a) p(a) ---> col 3
# h(c) p(c) ---> col 4

# p = particle state     == even integer
# h = hole state         == odd integer
# c = creation stae      == + ve
# a = annihilation stae  == - ve
 
# contraction rules: odd-odd, even-even, opposit signs(+- or -+)

import math
import os
import sys
import numpy as np
from itertools import permutations
from itertools import combinations

class op_ver:
    def __init__(self,ordr,sl,op):
        self.op=op
        self.sl=sl
        self.ordr=ordr
#        print("order",ordr)
    def vertices(self):
        opver=list()
        for i in range(self.ordr):
#            print(i,self.ordr)
            opver.append(self.op[i]+4*(self.sl-1))
        return(opver)

class gen_oper:
    def __init__(self,op,ordr,rep_int):
        self.op=op
        self.ordr=ordr
        self.rep_int=rep_int
    def operator(self):
        ava_int=[] # list of integers available for this operator
        oper_rep=[] # operator made of integers from available list
        for i in range(self.rep_int,rep_int+2*self.ordr):
            ava_int.append(i)
            ava_int.append(-i)
        c=len(ava_int)
 #       print(self.op,ava_int,c)
        for i in range(ordr):
            d=op[i]
            if d>4:
                d=d%4
#            print('d=',d)
            if d==1:
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]>0 and f==1.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]<0 and f==1.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break

            elif d==2:
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
#                    print('f',f,e,ava_int[j])
                    if ava_int[j]>0 and f==0.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]<0 and f==0.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break
                print('oper_rep,ava_int',oper_rep,ava_int)
            elif d==3:
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]<0 and f==1.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]<0 and f==0.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break

#                print(oper_rep,ava_int)
            elif d==4:
                e=len(ava_int)
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]>0 and f==1.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break
                for j in range(e):
                    f=math.sqrt(ava_int[j]*ava_int[j])%2
                    if ava_int[j]>0 and f==0.0:
                        oper_rep.append(ava_int[j])
                        del ava_int[j]
                        break

        return(oper_rep)
#        return(ava_int_p, ava_int_n)

class gen_comb:
    def __init__(self,rows):
        self.rows=rows
        print('self.rows',self.rows)
    def combination(self):
        number_pairs = []
        column_sum=list()
        column_mult=list()
        for i in range(len(self.rows)):
            for j in range(i + 1, len(self.rows)):
                pair_combinations = [(x, y, (i+1)**2 + (j+1)**2) for x in self.rows[i] for y in self.rows[j]]
### pair combination is necessary when there are more than two operators presents. 
                number_pairs.extend(pair_combinations)

        column_sum = np.sum(np.array(number_pairs)[:, [0, 1]], axis=1)
        for i in range(len(number_pairs)-1,-1,-1):
            if column_sum[i]%2 != 0 :
               del number_pairs[i]
        column_mult = np.array(number_pairs)[:, 0]*np.array(number_pairs)[:, 1]
#        print(column_mult,len(column_mult))
        for i in range(len(number_pairs)-1,-1,-1):
            if column_mult[i]>0:
               del number_pairs[i]
        return(number_pairs)


check_list=[]
cont_dia=[]
check_tuple=()

def initialization(n):
#    print('n*2-2,n-1',n*2-2,n-1)
    global check_list, cont_dia
    check_list=check_list[:n*2-2]
    cont_dia=cont_dia[:n-1]
    return(check_list,cont_dia)


class contraction:
    def __init__(self,sorted_list,max_con_pairs):
        self.sorted_list=sorted_list
        self.max_con_pairs=max_con_pairs
        print('self.sorted_list',self.sorted_list)
    def loop(self):
        all_con_diagrams=[]
#        print('self.max_con_pairs',self.max_con_pairs)
        global check_list, cont_dia
        for i1 in range(len(self.sorted_list)):
            initialization(1)
            x,y=self.sorted_list[i1]
            check_tuple=(x,y)
            check_list.append(x)
            check_list.append(y)
            cont_dia.append(check_tuple)
            l=len(cont_dia)
#            print('check_list,cont_dia,l,1',check_list,cont_dia,l)
            if l==self.max_con_pairs:
                all_con_diagrams.append(cont_dia)
                continue
            for i2 in range(i1+1,len(self.sorted_list)):
                initialization(2)
                x,y=self.sorted_list[i2]
                check_tuple=(x,y)
                if x in check_list:
                    continue
                if y in check_list:
                    continue
                check_list.append(x)
                check_list.append(y)
                cont_dia.append(check_tuple)
                l=len(cont_dia)
#                print('check_list,cont_dia,2',check_list,cont_dia)
                if l==self.max_con_pairs:
                    all_con_diagrams.append(cont_dia)
                    continue
                for i3 in range(i2+1,len(self.sorted_list)):
                    initialization(3)
                    x,y=self.sorted_list[i3]
                    check_tuple=(x,y)
                    if x in check_list:
                        continue
                    if y in check_list:
                        continue
                    check_list.append(x)
                    check_list.append(y)
                    cont_dia.append(check_tuple)
                    l=len(cont_dia)
 #                   print('check_list,cont_dia,3',check_list,cont_dia)
                    if l==self.max_con_pairs:
                        all_con_diagrams.append(cont_dia)
                        continue
                    for i4 in range(i3+1,len(self.sorted_list)):
                        initialization(4)
                        x,y=self.sorted_list[i4]
                        check_tuple=(x,y)
                        if x in check_list:
                            continue
                        if y in check_list:
                            continue
                        check_list.append(x)
                        check_list.append(y)
                        cont_dia.append(check_tuple)
                        l=len(cont_dia)
  #                      print('check_list,cont_dia,4',check_list,cont_dia)
                        if l==self.max_con_pairs:
                            all_con_diagrams.append(cont_dia)
                            continue
                        for i5 in range(i4+1,len(self.sorted_list)):
                            initialization(5)
                            x,y=self.sorted_list[i5]
                            check_tuple=(x,y)
                            if x in check_list:
                                continue
                            if y in check_list:
                                continue
                            check_list.append(x)
                            check_list.append(y)
                            cont_dia.append(check_tuple)
                            l=len(cont_dia)
   #                         print('check_list,cont_dia,4',check_list,cont_dia)
                            if l==self.max_con_pairs:
                                all_con_diagrams.append(cont_dia)
                                continue
                            for i6 in range(i5+1,len(self.sorted_list)):
                                initialization(6)
                                x,y=self.sorted_list[i6]
                                check_tuple=(x,y)
                                if x in check_list:
                                    continue
                                if y in check_list:
                                    continue
                                check_list.append(x)
                                check_list.append(y)
                                cont_dia.append(check_tuple)
                                l=len(cont_dia)
    #                            print('check_list,cont_dia,4',check_list,cont_dia)
                                if l==self.max_con_pairs:
                                    all_con_diagrams.append(cont_dia)
                                    continue
                                for i7 in range(i6+1,len(self.sorted_list)):
                                    initialization(7)
                                    x,y=self.sorted_list[i7]
                                    check_tuple=(x,y)
                                    if x in check_list:
                                        continue
                                    if y in check_list:
                                        continue
                                    check_list.append(x)
                                    check_list.append(y)
                                    cont_dia.append(check_tuple)
                                    l=len(cont_dia)
     #                               print('check_list,cont_dia,4',check_list,cont_dia)
                                    if l==self.max_con_pairs:
                                        all_con_diagrams.append(cont_dia)
                                        continue
                                    for i8 in range(i7+1,len(self.sorted_list)):
                                        initialization(8)
                                        x,y=self.sorted_list[i8]
                                        check_tuple=(x,y)
                                        if x in check_list:
                                            continue
                                        if y in check_list:
                                            continue
                                        check_list.append(x)
                                        check_list.append(y)
                                        cont_dia.append(check_tuple)
                                        l=len(cont_dia)
      #                                  print('check_list,cont_dia,4',check_list,cont_dia)
                                        if l==self.max_con_pairs:
                                            all_con_diagrams.append(cont_dia)
                                            continue
                                        for i9 in range(i8+1,len(self.sorted_list)):
                                            initialization(9)
                                            x,y=self.sorted_list[i9]
                                            check_tuple=(x,y)
                                            if x in check_list:
                                                continue
                                            if y in check_list:
                                                continue
                                            check_list.append(x)
                                            check_list.append(y)
                                            cont_dia.append(check_tuple)
                                            l=len(cont_dia)
       #                                     print('check_list,cont_dia,4',check_list,cont_dia)
                                            if l==self.max_con_pairs:
                                                all_con_diagrams.append(cont_dia)
                                                continue
                                            for i10 in range(i9+1,len(self.sorted_list)):
                                                initialization(10)
                                                x,y=self.sorted_list[i10]
                                                check_tuple=(x,y)
                                                if x in check_list:
                                                    continue
                                                if y in check_list:
                                                    continue
                                                check_list.append(x)
                                                check_list.append(y)
                                                cont_dia.append(check_tuple)
                                                l=len(cont_dia)
        #                                        print('check_list,cont_dia,4',check_list,cont_dia)
                                                if l==self.max_con_pairs:
                                                    all_con_diagrams.append(cont_dia)
                                                    continue

#        print('all_con_diagrams',all_con_diagrams)
        return(all_con_diagrams)





class contracted_dia(contraction):
    def __init__(self,perm,      con_pairs,       nset_con,con_type):
                    # perm_op[i],contracted_pairs,nset_con,con_type)
        self.perm=perm
        self.con_pairs=con_pairs
        self.nset_con=nset_con
        self.con_type=con_type
    def select_cont(self):
        con_list=[]
        new_con_list=[]
        new_con_list_2=[]
        result=[]
        l=len(self.perm)
        comb=list(combinations(self.perm,2))
        for i in range(len(comb)):
            x,y=comb[i]
            z=x**2+y**2
#            print('i,x,y',i,x,y)
            con_list=self.con_pairs
#            print('con_list',con_list)
            for j in range(len(con_list)):
                l,m,n=con_list[j]
#                print(l,m,n,y-x,l+m)
                if n == z and (y-x)*(l+m)>0 :
                    new_con_list.append(con_list[j])
        new_con_list_2=[(x,y) for x,y,_ in new_con_list]
#        print('new_con_list_2',new_con_list_2)
        sorted_con_list = sorted(new_con_list_2, key=lambda coord: coord[0])
        print('sorted_con_list',sorted_con_list)
        max_con_pairs=(self.nset_con*2-self.con_type)/2
#            print('max_con_pairs',max_con_pairs,self.nset_con,self.con_type)
#        sys.exit()
        if len(sorted_con_list) >= max_con_pairs:
            contraction_instance = contraction(sorted_con_list,max_con_pairs)
            result = contraction_instance.loop()
            print('result',result)
        return(result)

#            print()
#            print('new_con_list_2',sorted_con_list)
#            print()
#
## Initialize a dictionary to store dynamically named lists
#            separated_lists = {}
#            current_x= None
#
## Group the coordinates based on x-coordinates
#            k=0
#            for x, y in sorted_con_list:
##                for k, coords in enumerate(sorted_con_list, start=1):
#                if x != current_x:
#                    k=k+1
#                    key_name = f"list_{k}"
#                    current_x = x
#                    separated_lists[key_name] = [(x,y)]
#                else:
#                    separated_lists[key_name].append((x,y))
#            print(separated_lists)
##            result=self.loop()
#            contraction_instance = contraction(sorted_con_list)
#            result = contraction_instance.loop()

        
#                print(separated_lists["list_1"],separated_lists["list_2"])
#                print(separated_lists["list_1"])
#                print(separated_lists["list_2"])
#            all_poss_perm=list(permutations(new_con_list_2))
#            print(all_poss_perm)
#            result=self.contraction.loop(new_con_list_2)


            






npt = input("please enter the number of operators in the contraction process")
print("thanks.. please put operators in their vertices representation \
please note 1 --> h(c)h(a); 2 --> p(c)p(a); 3 --> h(a)p(a) and \
4 --> h(c)p(c) || where, 'h' is hole; 'p' is particle; 'c' is creation \
and 'a' is annihilation")
print("")

nopt=int(npt)
rep_int=1
operators=[]
op_ver_name=list()
op_rows=list()
contracted_pairs=list()
op_dictionary = {}
nop=list()
cont_diagrams=list()
nset_con=0 # number of contracted sets 
for i in range(nopt):
    nop.append(i+1)
    input_str = input(f"Enter {i+1}th operator vertices from left to right \
separated by spaces: ")
    op_ver_name = f"op_ver_{i+1}"
    operators.append(op_ver_name)
    op = list(map(int, input_str.split()))
    ordr=len(op)
#    print('ordr',ordr)
    a=op_ver(ordr,i+1,op)
    op_ver_name=a.vertices()
    print(operators[i],"==",op_ver_name,ordr)
    b=gen_oper(op_ver_name,ordr,rep_int)
    op_rows.append(b.operator())
    op_dictionary[f"op_{i+1}"] = b.operator()
    rep_int=rep_int+2*ordr
    nset_con=nset_con+ordr
op_matrix = [[key] + value for key, value in op_dictionary.items()]
ab = input('please specify what type of contraction you want! \
put 0: for full contraction, 1: for one pair of uncontracted arms, 2: \
for two paires of uncontracted arms and so on')
con_type=int(ab)
#for row in op_matrix:
#    print(row)
#print(op_matrix[0][0],op_rows)
c=gen_comb(op_rows)
contracted_pairs=c.combination()
print('contracted_pairs',contracted_pairs)

perm_op=list(permutations(nop))
print('nop,perm_op',nop,perm_op)
for i in range(len(perm_op)):
    print('iiii**************IIIIIIIIII',i,perm_op[i])
    d=contracted_dia(perm_op[i],contracted_pairs,nset_con,con_type)
    cont_diagrams=d.select_cont()
    print(cont_diagrams)
