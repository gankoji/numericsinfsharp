// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System

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

[<EntryPoint>]
let main argv =
    problem_24_3()

    0 // return an integer exit code