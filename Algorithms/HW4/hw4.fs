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

let findAdjacents node (adjMat:array<int> [,]) =
    let adjList = (Array.toList adjMat.[node,*])
    let rec searchList list acc =
        match list with
            a::b -> if (a > 0) then
                        acc::(searchList b (acc + 1))
                    else
                        (searchList b (acc + 1))
          | _ -> []

    printfn "%A" adjList
    //(searchList adjList 0)

[<EntryPoint>]
let main argv =
    // Problem 23.2: Monte-Carlo sampling of
    // an arbitrary distribution
    printfn "%A" (monteCarloSamples 50)
    // While the code is somewhat different (recursion vs looping)
    // They're very similar lengths: 25 lines in python, 21 in F#

    // Next up: 24.1, a mouse's random walk 
    printfn "%A" adjacencyMatrix.[1,*]
    0 // return an integer exit code
