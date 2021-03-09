// Learn more about F# at http://fsharp.org

open System
open FSharp.Plotly

// Linear Regression
let solveLinearLeastSquares (xs:list<double>) (ys:list<double>) =
    let n = Seq.length xs
    if not (n = (Seq.length ys)) then None
    else
        let xbar = Seq.average(xs)
        let ybar = Seq.average(ys)
        let xdiff = Seq.map (fun x -> x - xbar) xs
        let ydiff = Seq.map (fun y -> y - ybar) ys
        let xdiff2 = Seq.map (fun x -> x**2.0) xdiff
        let xdyd = Seq.zip xdiff ydiff |> Seq.map (fun (x,y) -> x*y)
        let yds = Seq.sum xdyd
        let x2ds = Seq.sum xdiff2
        let m = yds/x2ds
        let b = (Seq.sum(ys) - (m*Seq.sum(xs)))/(double n)
        Some(m, b)

// Gradient Descent w/ Backtracking
let gradientDescent f gradF x0 =
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

    descentLoop x0 6000.0 0.01

// Objective function for question 2
let f2 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    y**2.0 + (x - 10.0)**2.0 + 40.0*Math.Sqrt(-(x**2.0) + y**2.0 + Math.Sqrt((-(x**2.0) + y**2.0)**2.0 + 1.0))

// Gradient for q2
let g2 (p:seq<double>) = 
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let sub1 = (-(x**2.0) + y**2.0)
    let a = 
        (2.0*x + 40.0*(-x*sub1/Math.sqrt(sub1**2.0 + 1.0) - x)/Math.Sqrt(sub1 + Math.Sqrt(sub1**2.0 + 1.0)) - 20.0)
    let b = 
        (2.0*y + 40.0*(y*sub1/Math.Sqrt(sub1**2.0 + 1.0) + y)/Math.Sqrt(sub1 + Math.Sqrt(sub1**2.0 + 1.0)))

    [a; b]

// Objective function for q3
let f3 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    (x+3.0)**2.0 + (y**2.0)*Math.Exp(-2.0*x)

// Gradient for q3
let g3 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a =
        (2.0*x - 2.0*(y**2.0)*exp(-2.0*x) + 6.0)
    let b =
        (2.0*y*exp(-2.0*x))
    [a; b]

// Objective function for q4
let f4 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    (y**2.0)*exp(-2.0*x)/400.0 + (x + 3.0)**2.0
    
// Gradient for q4
let g4 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a =
        (2.0*x - 2.0*(y**2.0)*exp(-2.0*x)/200.0 + 6.0)
    let b =
        (2.0*y*exp(-2.0*x)/200.0)
    [a; b]

// Pretty Printing for GradDesc probs
let printGDResult f p (qnum:string) =
    let v = (f p)
    printfn "Question %s - Gradient Descent: %A" qnum f
    printfn "Optimal Point: %A" p
    printfn "Optimal Function Value: %A" v

[<EntryPoint>]
let main argv =

    // Question 1
    let xs = [-2.; -1.; 0.; 2.; 3.]
    let ys = [-3.; -1.; 5.; 5.; 1.]
    let q1 = solveLinearLeastSquares xs ys
    printfn "Question 16.2 - Linear Regression: %A" q1


    // Question 2
    printfn "%A" (f2 [-50.0; 40.0])
    let optPoint = gradientDescent f2 g2 [-50.0; 40.0]
    printGDResult f2 optPoint "17.1a"

    // Question 3
    printfn "17.1b Start: %A" (f3 [0.0; 1.0])
    let optPoint = gradientDescent f3 g3 [0.0; 1.0]
    printGDResult f3 optPoint "17.1b"

    // Question 4
    printfn "17.1c Start: %A" (f4 [0.0; 20.0])
    let optPoint = gradientDescent f4 g4 [0.0; 20.0]
    printGDResult f4 optPoint "17.1c"
    0 // return an integer exit code
