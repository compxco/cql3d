#!/usr/bin/python
###########################################################################
## script for converting CQL3D files into parallel form
## usage: doparallel.py in.f out.f insertions.f
## files: patterns.mpi - strings for cleaning (like "write(*,*)")
##        block.mpi    - list of output modules for blocking
## YuP[07-2016] removed usage of block.mpi as it was inserting lines at wrong place
##        dec.mpi      - list of fortran declaration insructions
###########################################################################


import sys, string

inf = []  ## input lines
out = []  ## output lines

###########################################################################
## clearing all terminal outputs (inf->out)
###########################################################################

pat = [] ## pattern for cleaning
buf = [] ## buffer for continuation lines
j_line = '' ## joined line

## clean single line 
def cleans(s):
    if s=='':
        return ''
    if s[0] in 'cC!':
        return ''
    state = 0
    r = s[:6]
    for c in s[6:]:
        if state==0:
            if c=='"':
                state = 1
            elif c=="'":
                state = 2
            elif c=='!':
                break
            else:
                r += c
        elif state==1:
            if c=='"':
                state = 0
        elif state==2:
            if c=="'":
                state = 0
    if s.strip=='':
        return ''
    if len(r)<6:
        r += ' '*6
    return r

## saving buffer data into out list (c - prefix)
def printbuf(c=''):
    global buf, out
    if c!='' and buf[0][:6].strip()!='':
        ## it is labeled line and it should be commented, so we use new line
        ## with given label and CONTINUE instruction
        out.append(buf[0][:6]+'CONTINUE !***')
    for b in buf:
        if len(b)>0 and b[0] in 'cC':
            out.append(b)
        else:
            out.append(c+b)

## processing buffer data 
def procbuf():
    if buf==[]:
        return
    s = j_line.strip()
    for p in pat:
        if p in s:
            printbuf('CMPI :::')
            return
    printbuf()

## main cleaning module
def clean():
    global buf, j_line, pat
    ## reading patterns for cleaning
    P = open('mpi/patterns.mpi','r').readlines()
    for p in P:
        pat.append(p.strip())
    ## start processing
    buf = []
    for s in inf:
        ss = cleans(s)
        if ss=='': ## empty or comment line
            buf.append(s)
        elif ss[5]!=' ': ## continuation line
            buf.append(s)
            j_line += ss[6:]
            continue
        else: ## normal line
            procbuf()
            buf = [s]
            j_line = ss
    procbuf()

###########################################################################
## blocking all outputs from working processes (out->out)
###########################################################################

## detecting endpoint of declaration
def start(s, D):
    if s=='':
        return False
    if s[0] in 'cC!':
        return False
    if len(s)>5 and s[5]!=' ':
        return False
    c = s[6:].strip()
    if len(c)==0:
        return False
    pos = 0
    com = ''
    while c[pos] in string.ascii_letters:
        com += c[pos]
        pos += 1
        if pos==len(c):
            break
    if com.lower() in D:
        return False
    return True

## main blocking module
## YuP[07-2016] removed usage of block.mpi as it was inserting lines at wrong place.
## YuP: Instead, use appropriate lines from mpins_par.f 
##     (such as CMPIINSERT_IF_RANK_NE_0_RETURN) in subroutines which are
##     supposed to be called by mpirank=0 only.
def block():
    ## blocking lines
    ins = ' '*6+"include 'mpilib.h'\n"+' '*6+"if(mpirank.ne.0) return"
    ## checking that current file is in the list of blocked files
    check = False
    for l in open('mpi/block.mpi','r').readlines():
        if l.strip()==sys.argv[1]:
            check = True
            break
    if not check: ## current file shouldn't be blocked
        return
    ## reading delcaration instructions for detecting inserting point
    D = open('mpi/dec.mpi','r').readlines()
    for i in range(len(D)):
        D[i] = D[i].strip()
    ## searching insertion point and inserting blocking instruction
    state = 0
    for i in range(len(out)):
        if state == 0 and start(out[i],D):
            out.insert(i,ins)
            break

###########################################################################
## inserting mpi instructons (out->inf)
###########################################################################

## main inserting module
def dompi():
    global out, inf
    ins = {}
    inf = []
    ## reading insertions
    L = open(sys.argv[3],'r').readlines()
    for l in L:
        if l[0]=='C':
            key = l.strip()
            ins[key] = []
        else:
            ins[key].append(l[:-1])
    ## searching and inserting 
    mode = ''
    for l in out:
        if mode=='replace':
            inf.append('CMPI :::'+l)
            mode = ''
            continue
        ll = l.strip()
        if l[0:4]=='CMPI' and ll in ins:
            inf.append('CMPI >>>')
            for i in ins[ll]:
                inf.append(i)
            inf.append('CMPI <<<')
            if l[0:11]=='CMPIREPLACE':
                mode = 'replace'
            continue
        inf.append(l)

###########################################################################
## chief function
###########################################################################

def main():
    global inf, out

    ## reading input file into inf buffer
    if "mpilib.f" in sys.argv[1]:
        # Don't process this source file as its contents will be parsed
        # into others
        return
    f = open(sys.argv[1],'r')
    s = f.readline()
    while s!='':
        inf.append(s.rstrip())
        s = f.readline()
    f.close()
    
    ## doing parallelizing    
    clean() ## clear all outputs on terminal (write(*,*) and so on)
    block() ## block all outputs for workers 
    dompi() ## inserting mpi instructions into code

    ## writing result
    f = open(sys.argv[2],'w')
    for s in inf:
        f.write(s + "\r\n")
    f.close()

## do all
main()
