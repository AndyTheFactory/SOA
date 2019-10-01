import random
import math
import numpy as np

MAX_NUMBER=100
NR_AGENTS=10
sieve=list(range(2,MAX_NUMBER+1))
sieve_trace=list([0]*(MAX_NUMBER-1))
agents=[]

def main():

    for i in range(0,NR_AGENTS):
        agents.append(Agent(random.randint(1, int(math.sqrt(MAX_NUMBER)-1))))

    while not isReady():
        print_sieve()
        for ag in agents:
            ag.tick()
    print_primes()

def isReady():
    return np.prod(sieve_trace)>0

def check_agent(n):
    res=False
    for ag in agents:
        if ag.current_nr==n:
            res=True
            break
    return res

def print_sieve():
    s="1"
    for ix,nr in enumerate(sieve):
        if check_agent(ix):
            s+="A"
        else:
            if (nr>0):
                s+="*"
            else:
                s+="-"
    print(s)
def print_primes():
    s="1,"
    for i in sieve:
        if i>0 :
            s+="%d," % i
    print(s)
class Agent:
    current_nr=0
    start_nr=0
    increment=0
    def __init__(self,startnr):
        while check_agent(startnr):
            startnr+=1
            if startnr>math.sqrt(MAX_NUMBER):
                startnr=0
        self.startfrom(startnr)
    def startfrom(self,nr):
        if nr>MAX_NUMBER-2:
            nr=random.randint(1, int(math.sqrt(MAX_NUMBER)-1))
        self.start_nr=nr
        self.current_nr=nr
        self.increment=sieve[nr]
        sieve_trace[nr]=1

    def stepin(self):
        sieve[self.current_nr] = 0
        sieve_trace[self.current_nr]=1

    def tick(self):
        #  Make move
        if self.start_nr>MAX_NUMBER-2:
            # I am done
            return
        if self.start_nr==self.current_nr and sieve[self.current_nr]==0:
            self.startfrom(self.start_nr+1)
        else:
            self.current_nr += self.increment  # Step in multiples

        if self.current_nr>MAX_NUMBER-2:
            self.startfrom(self.start_nr+1)
        else:
            if self.start_nr != self.current_nr:
                self.stepin()

if __name__ == '__main__':
    main()