// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System

// Define a function to construct a message to print
let from whom =
    sprintf "from %s" whom

let f181 (p:seq<double>) = 
    let x = Seq.item 0 p
    let y = Seq.item 1 p

    let a = Math.Exp(8.0*x - 13.0*y - 21.0)
    let b = Math.Exp(21.0*y - 13.0*x - 34.0)
    let c = 0.0001*Math.Exp(x+y)

    a + b + c

let g181 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a = Math.Exp(8.0*x-13.0*y+21.)
    let b = Math.Exp(21.0*y-13.0*x-34.)
    let c = 0.0001*Math.Exp(x+y)

    [8.0*a-13.0*b+c; -13.0*a+21.0*b+c]

// Gradient Descent w/ Backtracking
let gradientDescent f gradF x0 f0=
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

    descentLoop x0 f0 0.01

[<EntryPoint>]
let main argv =
    let message = from "F#" // Call the function
    printfn "Hello world %s" message
    printfn "Function result: %f" (f181 [0.;0.])
    let fstar = gradientDescent f181 g181 [0.;0.] 6000.
    printfn "Optimization result: %A" fstar
    0 // return an integer exit code