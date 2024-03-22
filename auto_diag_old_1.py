# h(c) h(a) ---> col 1
# p(c) p(a) ---> col 2
# h(a) p(a) ---> col 3
# h(c) p(c) ---> col 4
import math
import numpy as np
from itertools import permutations
from itertools import combinations

class op_ver:
    def __init__(self,ordr,sl,op):
        self.op=op
        self.sl=sl
        self.ordr=ordr
        print("opr",ordr)
    def vertices(self):
        opver=list()
        for i in range(self.ordr):
            print(i,self.ordr)
            opver.append(self.op[i]+4*(self.sl-1))
        return(opver)

class gen_oper:
    def __init__(self,op,ordr,rep_int):
        self.op=op
        self.ordr=ordr
        self.rep_int=rep_int
    def operator(self):
        ava_int=[]
        oper_rep=[]
        for i in range(self.rep_int,rep_int+2*self.ordr):
            ava_int.append(i)
            ava_int.append(-i)
        c=len(ava_int)
        print(self.op,ava_int,c)
        for i in range(ordr):
            d=op[i]
            if d>4:
                d=d%4
            print('d=',d)
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
                print(oper_rep,ava_int)
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

#        print(ava_int_n)
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
                number_pairs.extend(pair_combinations)

        column_sum = np.sum(np.array(number_pairs)[:, [0, 1]], axis=1)
        print(number_pairs,len(number_pairs))
        print()
        print(column_sum,len(column_sum))
        for i in range(len(number_pairs)-1,-1,-1):
            if column_sum[i]%2 != 0 :
               del number_pairs[i]
        column_mult = np.array(number_pairs)[:, 0]*np.array(number_pairs)[:, 1]
        print(column_mult,len(column_mult))
        for i in range(len(number_pairs)-1,-1,-1):
            if column_mult[i]>0:
               del number_pairs[i]
        return(number_pairs)


check_list=[]
cont_dia=[]
check_tuple=()
def initialization(n):
    ll1=len(check_list)
    ll2=len(cont_dia)
    if ll1 >= n:
        for i in range(n:ll1):
            del check_list[i]
            del cont_dia[i]

class contraction:
    def __init__(self,separated_lists):
        self.separated_lists=separated_lists
    def loop(self):
        list_1=self.separated_lists.get('list_1',[])
        list_2=self.separated_lists.get('list_2',[])
        list_3=self.separated_lists.get('list_3',[])
        list_4=self.separated_lists.get('list_4',[])
        list_5=self.separated_lists.get('list_5',[])
        print(list_1,list_2,list_3,list_4,list_5)
        for i1 in range(len(list_1)):
            x,y=list_1[i1]
            check_tuple=(x,y)
            check_list.append(y)
            cont_dia.append(check_tuple)
            for i2 in range(len(list_2)):
                initialization(2)
                x,y=list_2[i2]
                check_tuple=(x,y)
                if y in check_list:
                    continue
                check_list.append(y)
                cont_dia.append(check_tuple)
                for i3 in range(len(list_3)):
                    x,y=list_3[i3]
                    check_tuple=(x,y)
                    if y in check_list:
                        continue
                    check_list.append(y)
                    cont_dia.append(check_tuple)
                    for i4 in range(len(list_4)):
                        x,y=list_4[i4]
                        check_tuple=(x,y)
                        if y in check_list:
                            continue
                        check_list.append(y)
                        cont_dia.append(check_tuple)
                        for i5 in range(len(list_5)):
                            x,y=list_5[i5]
                            check_tuple=(x,y)
                            if y in check_list:
                                continue
                            check_list.append(check_tuple)
                            cont_dia.append(check_tuple)

        print('cont_dia',cont_dia)





class contracted_dia(contraction):
    def __init__(self,perm,con_pairs):
        self.perm=perm
        self.con_pairs=con_pairs
    def select_cont(self):
        con_list=[]
        new_con_list=[]
        new_con_list_2=[]
        l=len(self.perm)
        comb=list(combinations(self.perm,2))
        for i in range(len(comb)):
            x,y=comb[i]
            z=x**2+y**2
            print(x,y)
            con_list=self.con_pairs
            for j in range(len(con_list)):
                l,m,n=con_list[j]
#                print(l,m,n,y-x,l+m)
                if n == z and (y-x)*(l+m)>0 :
                    new_con_list.append(con_list[j])
                new_con_list_2=[(x,y) for x,y,_ in new_con_list]
            sorted_con_list = sorted(new_con_list_2, key=lambda coord: coord[0])
            print()
            print('new_con_list_2',sorted_con_list)
            print()

# Initialize a dictionary to store dynamically named lists
            separated_lists = {}
            current_x= None

# Group the coordinates based on x-coordinates
            k=0
            for x, y in sorted_con_list:
#                for k, coords in enumerate(sorted_con_list, start=1):
                if x != current_x:
                    k=k+1
                    key_name = f"list_{k}"
                    current_x = x
                    separated_lists[key_name] = [(x,y)]
                else:
                    separated_lists[key_name].append((x,y))
            print(separated_lists)
#            result=self.loop()
            contraction_instance = contraction(separated_lists)
            result = contraction_instance.loop()

        
#                print(separated_lists["list_1"],separated_lists["list_2"])
#                print(separated_lists["list_1"])
#                print(separated_lists["list_2"])
#            all_poss_perm=list(permutations(new_con_list_2))
#            print(all_poss_perm)
#            result=self.contraction.loop(new_con_list_2)
#            return(result)


            






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
for i in range(nopt):
    nop.append(i+1)
    input_str = input(f"Enter {i+1}th operator vertices from left to right \
separated by spaces: ")
    op_ver_name = f"op_ver_{i+1}"
    operators.append(op_ver_name)
    op = list(map(int, input_str.split()))
    ordr=len(op)
    print('ordr',ordr)
    a=op_ver(ordr,i+1,op)
    op_ver_name=a.vertices()
    print(operators[i],"=",op_ver_name,ordr)
    b=gen_oper(op_ver_name,ordr,rep_int)
    op_rows.append(b.operator())
    op_dictionary[f"op_{i+1}"] = b.operator()
    rep_int=rep_int+2*ordr
op_matrix = [[key] + value for key, value in op_dictionary.items()]
for row in op_matrix:
    print(row)
print(op_matrix[0][0],op_rows)
c=gen_comb(op_rows)
contracted_pairs=c.combination()
print(contracted_pairs)

perm_op=list(permutations(nop))
print(nop,perm_op)
for i in range(len(perm_op)):
    d=contracted_dia(perm_op[i],contracted_pairs)
    cont_diagrams=d.select_cont()
    print('iiii',i)
   
