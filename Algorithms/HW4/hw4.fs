// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

module Homework4

open System
open System.IO

let f232 x =
    if (x > 0.) && (x <= 2.) then
        if (x <=1.) then
            x**2.
        else
            Math.Sqrt(2. - x)
    else
        0.

// Next we need to do the sampling
// Process is the same as the python code
// Sample two random numbers. If the second is less than
// the 'sampled' function, evaluated at the first number, we accept the first 
// number as the sample of the distribution we're after. 
// If we don't accept it, we recurse until we get a suitable sample. 
let rand = Random()
let rec oneSample ()=
    let u1 = rand.NextDouble()*2.
    let u2 = rand.NextDouble()
    if u2 <= (f232 u1) then
        u1
    else (oneSample ())


let monteCarloSamples (nSamples:int) =
    // Use oneSample above to get to nSamples
    let samples = []
    let rec sample samples acc =
        if acc = nSamples then
            samples
        else
            (sample (List.append [(oneSample ())] samples) (acc + 1))
    (sample samples 0)

// So for 24.1, the question becomes one of keeping the original method
// or figuring out a slightly more efficient one. Let's start with the
// original method.
let adjacencyMatrix = array2D [[0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
                               [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
                               [0; 1; 0; 1; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
                               [0; 0; 1; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
                               [1; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
                               [0; 0; 0; 0; 1; 0; 1; 0; 0; 1; 0; 0; 0; 0; 0; 0];
                               [0; 0; 1; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
                               [0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
                               [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
                               [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
                               [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
                               [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 1; 0; 0; 0; 0; 1];
                               [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 1; 0; 0];
                               [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 1; 0; 1; 0];
                               [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 1];
                               [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 1; 0]]

let findAdjacents node (adjMat:int [,]) =
    let adjList = (Array.toList adjMat.[node,*])
    let rec searchList (list:List<int>) acc =
        match list with
            a::b -> if (a > 0) then
                        acc::(searchList b (acc + 1))
                    else
                        (searchList b (acc + 1))
          | _ -> []

    (searchList adjList 0)

let chooseMove (adjacent:List<int>) =
    let n = adjacent.Length
    let u = rand.Next(n)
    List.item u adjacent

let randomWalkOnGraph start (adjMat:int [,]) =
    let special = [0;1]
    let rec walkGraph current =
        let adjacents = (findAdjacents current adjMat)
        let nextNode = (chooseMove adjacents)
        if (List.contains nextNode special) then
            if (nextNode = 0) then
                1
            else
                0
        else
            (walkGraph nextNode)
    (walkGraph start)

let monteCarloWalk (nSamples:int) =
    let samples = []
    let rec sample samples acc =
        if acc = nSamples then
            samples
        else
            (sample (List.append [(randomWalkOnGraph 15 adjacencyMatrix)] samples) (acc + 1))
    (sample samples 0)

let brownianMotion () =
    let rec walk x y =
        let u = 0.1*rand.NextDouble()
        let xp = 2.*(rand.NextDouble() - 0.5)
        let yp = 2.*(rand.NextDouble() - 0.5)
        let mag = sqrt(xp**2. + yp**2.)
        let xn = x + u*xp/mag
        let yn = y + u*yp/mag
        if xn < 0. then
            0
        else if xn > 1. then
            1
        else if yn < 0. then
            2
        else if yn > 1. then
            3
        else 
            (walk xn yn)

    (walk (1./3.) (1./6.))

let updateElement key f st =
    st |> List.map (fun (k, v) -> if k = key then k, f v else k, v)

// Helper functions for 25.1
let p x =
    if ((x >= 0.) && (x < 100.)) then
        Math.Exp(-x)
    else
        0.

let a x xp =
    let p1 = (p x)
    let p2 = (p xp)
    Math.Min(1., p2/p1)

let expectation (chain:List<float>) =
    let rec sumdiffs chain acc =
        match chain with
        | a::(b::c) -> (sumdiffs (b::c) ((a-1.)*(b-1.) + acc))
        | _ -> acc

    let res = (sumdiffs chain 0.)
    (float res)/(float chain.Length)

let markovChain s x0 =
    let nsteps = (int 1e6)
    let rec chainStep (xs:List<float>) acc =
        if acc > nsteps then
            xs
        else
            let x = (List.head xs) 
            let u = s*(rand.NextDouble() - 0.5)
            let xp = x + u
            let A = (a x xp)
            let u = rand.NextDouble()
            if A >= u then
                (chainStep (xp::xs) (acc+1))
            else
                (chainStep (x::xs) (acc+1))

    (expectation (chainStep [x0] 0))

let problem1 () = 
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let samples = (monteCarloSamples (int 1e4))
    stopwatch.Stop()
    printfn "Problem 1 Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

let problem2 () =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let nRuns = (int 1e6)
    let successes = (List.sum (monteCarloWalk nRuns))
    printfn "Probability that mouse survivies: %A" ((float successes)/(float nRuns))
    stopwatch.Stop()
    printfn "Problem 2 Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

let problem3 () = 
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let ntrials = (int 1e7)
    let dirsStart = [(0,0);(1,0);(2,0);(3,0)]
    let rec doTrials dirs acc =
        if acc > ntrials then
            dirs
        else
            let result = brownianMotion()
            doTrials (updateElement result (fun (x) -> x + 1) dirs) (acc + 1)
    
    let outcomes = (doTrials dirsStart 0)
    printfn "Result of Brownian Walk: %A" (outcomes |> List.map (fun (x,y) -> (float y)/(float ntrials)))
    stopwatch.Stop()
    printfn "Problem 3 Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

let problem4 () =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    let sigmas = [0.1;1.;2.;3.;4.9;5.;5.1; 5.2; 20.; 25.; 30.]
    let exs = sigmas |> List.map (fun (s) -> (s,(markovChain s 0.)))
    let bestEx = List.minBy (fun (a,b) -> b) exs
    printfn "Minimum expectation: %A, at sigma %A" (snd bestEx) (fst bestEx)
    stopwatch.Stop()
    printfn "Problem 4 Elapsed time: %f" stopwatch.Elapsed.TotalSeconds

[<EntryPoint>]
let main argv =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    // Problem 23.2: Monte-Carlo sampling of
    // an arbitrary distribution
    problem1()

    // Next up: 24.1, a mouse's random walk 
    problem2()

    // 24.5, Brownian Motion
    problem3()

    // 25.1, Autocorrelation of a Markov chain
    problem4()

    0 // return an integer exit code
