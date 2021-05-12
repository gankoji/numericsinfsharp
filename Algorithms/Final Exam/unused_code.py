## First, since we're forbidden to use a normal distribution sampler, we must build our own
def normalSample(n):
    if (n%2 == 1):
        # only support even numbers of samples, for simplicity
        n = n+1
    ## We use here the Box-Muller algorithm, as shown on pp 27 of the course notes
    ns = np.zeros((n,))
    for i in range(0,n//2):
        u = np.random.uniform(0,1,size=2)
        r = math.sqrt(-2.*math.log(u[0]))
        ns[2*i] = r*math.cos(2*math.pi*u[1])
        ns[2*i+1] = r*math.sin(2*math.pi*u[1])

    return ns
def verifyNormalGenerator():
    n = int(1e6)
    ns = normalSample(n)
    plt.hist(ns, bins=50,density=True)
    plt.title("Normal generator histogram")
    plt.xlabel("Sample value")
    plt.ylabel("Normalized frequency")
    plt.savefig("p5_nhist.png")

    ns = np.sort(ns)
    xs = np.linspace(0,1,n)
    plt.figure()
    plt.plot(ns,xs)
    plt.title("Normal generator CDF")
    plt.xlabel("Sample value")
    plt.ylabel("Cumulative Probability")
    plt.savefig("p5_ncdf.png")