// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open System.IO
open FSharp.Plotly
open MathNet.Numerics.LinearAlgebra

let t = 100000.
let newtonL2Eq f dF ddF xin =
    printfn "Equality Constrained Minimization"
    let dt0 = 0.01
    let rec descentLoop x (fval:double) dt = 
        let newf = (f x)
        let newg = (dF x)
        let newh = (ddF x)
        if (Math.Abs (newf - fval)) < 1e-10 then
            printfn "Delta: %A" (Math.Abs (newf - fval))
            x
        else
            let newx = Seq.zip x newg |> Seq.map (fun (x, dx) -> x - dt*dx)
            let inter = (f newx)
            if (inter > fval) then
                descentLoop x (fval + 1.0) (0.5*dt)
            else 
                let newt = 1.1*dt
                descentLoop newx newf newt

    descentLoop xin (f xin) dt0
    // x = xin
    // while True:
    //     F = f(x)
    //     DF = dF(x)
    //     if math.fabs(np.linalg.norm(DF)) < 1e-8:
    //         break
    //     DDF = ddF(x)
    //     if np.linalg.cond(DDF) < 1/sys.float_info.epsilon:
    //         d = -np.linalg.solve(DDF,DF)
    //     else:
    //         break
    //     flag = False
    //     while (f(x + d) >= F):
    //         d = 0.5*d
    //         if math.fabs(np.linalg.norm(d)) < 1e-6:
    //             if flag:
    //                 break
    //             d = -DF
    //             flag = True
    //     x = x + d
    // return x

// Gradient Descent w/ Backtracking
let gradientDescent f gradF x0 f0 dt0=
    let rec descentLoop x (fval:double) dt = 
        let newf = (f x)
        let newg = (gradF x)
        if (Math.Abs (newf - fval)) < 1e-10 then
            printfn "Delta: %A" (Math.Abs (newf - fval))
            x
        else
            let newx = Seq.zip x newg |> Seq.map (fun (x, dx) -> x - dt*dx)
            let inter = (f newx)
            if (inter > fval) then
                descentLoop x (fval + 1.0) (0.5*dt)
            else 
                let newt = 1.1*dt
                descentLoop newx newf newt

    descentLoop x0 f0 dt0

// Define a function to construct a message to print
let from whom =
    sprintf "from %s" whom

[<EntryPoint>]
let main argv =
    let message = from "F#" // Call the function
    printfn "Hello world %s" message
    0 // return an integer exit code