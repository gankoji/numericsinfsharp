// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System

let rand = Random()

// Credit to http://fssnip.net/o7/title/Simple-normally-distributed-random-number-generator
// for this implementation of the polar form of the Box-Muller transform
// to generate normally distributed samples
let normalDistRandom mean stdDev = 
    let rand = new System.Random()
    let rec polarBoxMullerDist () = seq {
            let rec getRands () =
                let u = (2.0 * rand.NextDouble()) - 1.0
                let v = (2.0 * rand.NextDouble()) - 1.0
                let w = u * u + v * v
                if w >= 1.0 then
                    getRands()
                else
                    u, v, w
            let u, v, w = getRands()
            
            let scale = System.Math.Sqrt(-2.0 * System.Math.Log(w) / w)
            let x = scale * u
            let y = scale * v
            yield mean + (x * stdDev); yield mean + (y * stdDev); yield! polarBoxMullerDist ()
        }
    polarBoxMullerDist ()

// First up: Problem 24.3: Importance Sampling
// The PDF of a normal variate
let normalPDF mu sigma x =
    (1./(sigma*sqrt(2.*Math.PI)))*Math.Exp(-0.5*((x - mu)/sigma)**2.)

// The sampling routine
let problem_24_3 () =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let n = int(1e5)

    // First, the naive approach (monte carlo)
    let stdNormals = Seq.take n (normalDistRandom 0. 1.)
    let getEV acc x =
        acc + (x**20.)/(float n)
    let mcev = stdNormals |> Seq.fold getEV 0.
    printfn "Naive MC Expected Value: %A" mcev

    // On to importance sampling
    let gpdf = normalPDF 0. 0.01
    let fpdf = normalPDF 0. 1.
    let gNormals = Seq.take n (normalDistRandom 0. 0.01)
    let scaledEV f g acc x =
        let scale = (f x)/(g x)
        acc + scale*(x**20.)/(float n)
    let isEV = scaledEV fpdf gpdf
    let isev = gNormals |> Seq.fold isEV 0.
    printfn "Importance Sampled Expected Value: %A" isev
    stopwatch.Stop()
    printfn "Problem 24.3 Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

// Problem 24.6: Markov Chains
let Ta (x:float, y:float):float =
    if y <= (x + 1.) then
        1./(x+2.)
    else
        0.

let factorial (x:float):float =
    let rec subfac x acc =
        if x < 2. then
            acc
        else
            subfac (x-1.) (acc*x)
    subfac x 1.

let econst:float = Math.Exp(-1.)
let pi0 (x:float):float =
    econst*float(factorial x)

// Acceptance probability function. Takes T and P as curried args
let a T P (x:float) (xp:float) =
    let newT:float = T(xp,x)
    let oldT:float = T(x,xp)
    if Math.Abs(newT) < 1e-4 then
        1.
    else if Math.Abs(oldT) < 1e-4 then
        0.
    else
        let newP:float = P(xp)
        let oldP:float = P(x)
        if Math.Abs(oldP) < 1e-4 then
            1.
        else
            Math.Max(1., (newP/oldP)*(oldT/newT))

// A generalized M-H MC generator
let markovChain s x0 nsteps a =
    let rec chainStep (xs:List<float>) acc =
        if acc > nsteps then
            xs
        else
            let x = (List.head xs) 
            let u = (Seq.head (normalDistRandom 0. s))
            let xp = x + u
            let A = (a x xp)
            let u = rand.NextDouble()
            if A >= u then
                (chainStep (xp::xs) (acc+1))
            else
                (chainStep (x::xs) (acc+1))

    (chainStep [x0] 0)
    
let problem_24_6_a () =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let nsamples = int(1e6)
    let samples = (markovChain 1. 0. nsamples (a Ta pi0))
    printfn "Sampling done"
    stopwatch.Stop()
    printfn "Problem 24.6a Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

// Problem 24.6b: Monte Carlo of an Integer Markov Chain
// Pb, Tb combine to make Ab, the acceptance probability
// of any given step in the chain
let Pb x = 
    1.

let Tb (x:int, y:int):float =
    if y <= (x + 1) then
        1./(float(x)+2.)
    else
        0.

let Ab (x:int) (xp:int) =
    let newT:float = Tb(x,xp)
    let oldT:float = Tb(xp,x)
    if Math.Abs(newT) < 1e-4 then
        1.
    else if Math.Abs(oldT) < 1e-4 then
        0.
    else
        let newP:float = Pb(xp)
        let oldP:float = Pb(x)
        if Math.Abs(oldP) < 1e-4 then
            1.
        else
            Math.Max(1., (newP/oldP)*(oldT/newT))

// markovChain2 builds the actual chain
// Sampling a random step, determining its probability
// of acceptance by Ab above. Builds from right to left, 
// then transforms to seq and reverses for original order.
let markovChain2 (x0:int) nsteps a =
    let rec chainStep (xs:List<int>) acc =
        if acc > nsteps then
            xs
        else
            let x = (List.head xs) 
            let xp = rand.Next(x+2)
            let A = (a x xp)
            let u = rand.NextDouble()
            if A >= u then
                (chainStep (xp::xs) (acc+1))
            else
                (chainStep (x::xs) (acc+1))

    (chainStep [x0] 0) |> List.toSeq |> Seq.rev

// Finally, actually sample the chains
// Each chain only has 15 steps, and we're interested in 
// whether or not the final state has a value of 10
let problem_24_6_b () =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let nsamples = 15
    let ntrials = int(2e8)
    let rec runTrials successes acc =
        if acc > ntrials then
            successes
        else
            let samples = (markovChain2 1 nsamples Ab)
            if (Seq.item 14 samples) = 10 then
                runTrials (successes + 1) (acc+1)
            else
                runTrials successes (acc+1)

    let totalS = (runTrials 0 0)
    printfn "Filtering done. Result: %A" (float(totalS)/float(ntrials))
    stopwatch.Stop()
    printfn "Problem 24.6b Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

[<EntryPoint>]
let main argv =
    problem_24_3()
    problem_24_6_a()
    problem_24_6_b()
    0 // return an integer exit code