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
        // if (nextNode in special)
        if (List.contains nextNode special) then
            if (nextNode = 0) then
                1
            else
                0
        else
            (walkGraph nextNode)
    (walkGraph start)

let monteCarloWalk (nSamples:int) =
    // Use oneSample above to get to nSamples
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

// Once we have that, we loop until we get one of the coordinates to exceed the boundaries
// Then, we just have to check which one, and monte carlo to find the probabilities
let updateElement key f st =
    st |> List.map (fun (k, v) -> if k = key then k, f v else k, v)

let problem3 = 
    let ntrials = (int 1e4)
    let dirsStart = [(0,0);(1,0);(2,0);(3,0)]
    let rec doTrials dirs acc =
        if acc > ntrials then
            dirs
        else
            let result = (brownianMotion())
            doTrials (updateElement result (fun (x) -> x + 1) dirs) (acc + 1)
    
    let outcomes = (doTrials dirsStart 0)
    outcomes |> List.map (fun (x,y) -> (float y)/(float ntrials))
[<EntryPoint>]
let main argv =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    // Problem 23.2: Monte-Carlo sampling of
    // an arbitrary distribution
    let samples = (monteCarloSamples (int 1e6))
    // While the code is somewhat different (recursion vs looping)
    // They're very similar lengths: 25 lines in python, 21 in F#

    // Next up: 24.1, a mouse's random walk 
    let nRuns = 100000
    let successes = (List.sum (monteCarloWalk nRuns))

    stopwatch.Stop()
    printfn "Probability that mouse survivies: %A" ((float successes)/(float nRuns))
    printfn "Result of Brownian Walk: %A" (problem3)
    printfn "Elapsed time: %f" stopwatch.Elapsed.TotalSeconds
    0 // return an integer exit code
