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

    // def randomWalkOnGraph(start, adjMat):
    //     terminated = False
    //     special = [0, 1] # Cheese is at node 0, cat at node 1
    //     current = start
    //     while (not terminated):
    //         adjacents = findAdjacents(current, adjMat)
    //         nextNode = chooseMove(adjacents)
    //         if (nextNode in special):
    //             terminated = True
    //             if nextNode == 0:
    //                 return 1
    //             else:
    //                 return 0
    //         current = nextNode

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

[<EntryPoint>]
let main argv =
    let stopwatch = System.Diagnostics.Stopwatch.StartNew()
    // Problem 23.2: Monte-Carlo sampling of
    // an arbitrary distribution
    printfn "%A" (monteCarloSamples (int 1e6))
    // While the code is somewhat different (recursion vs looping)
    // They're very similar lengths: 25 lines in python, 21 in F#

    // Next up: 24.1, a mouse's random walk 
    let nRuns = 1000000
    let successes = (List.sum (monteCarloWalk nRuns))

    stopwatch.Stop()
    printfn "Probability that mouse survivies: %A" ((float successes)/(float nRuns))
    printfn "Elapsed time: %f" stopwatch.Elapsed.TotalSeconds
    0 // return an integer exit code
