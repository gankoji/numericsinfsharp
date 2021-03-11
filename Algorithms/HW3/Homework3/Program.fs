// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open System.IO
open FSharp.Plotly
open MathNet.Numerics.LinearAlgebra

let t = 100000.
let newtonL2Eq f dF ddF (xin:Vector<double>) =
    let rec innerLoop (x:Vector<double>) (d:Vector<double>) f (fval:double) =
        if (f (x.Add d)) >= fval then 
            //printfn "Backing up"
            if (d.L2Norm() > 1e-6) then 
                (innerLoop x (0.5*d) f fval)
            else (x.Add d)
        else (x.Add d)

    let rec descentLoop x (fval:double) = 
        //printfn "Step"
        let newf = (f x)
        let math:Matrix<double> = (ddF x)
        let condh = try 
                        math.ConditionNumber()
                    with
                        | ex -> printfn "%A" ex; 1.1e9
        let vecg:Vector<double> = (dF x)
        let normg = vecg.L2Norm()
        if ((Math.Abs (newf - fval)) < 1e-6) || (normg < 1e-8) || (condh > 1e9) then
            printfn "Delta: %A" (Math.Abs (newf - fval))
            printfn "NormG: %A" normg
            printfn "CondH: %A" condh
            (CreateVector.DenseOfEnumerable x)
        else
            let d = math.Solve(vecg).Negate()
            if d.L2Norm() > 1e-8 then
                //let newx = Seq.zip x d |> Seq.map (fun (x, dx) -> x - dx)
                //printfn "%A" x
                //printfn "%A" d
                let nx = (CreateVector.DenseOfEnumerable x)
                let x = (innerLoop nx d f newf)
                //printfn "%A" x
                descentLoop x newf
            else
                (CreateVector.DenseOfEnumerable x)

    descentLoop xin ((f xin) + 10000.)

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

let f191 (p:Vector<double>) =
    let x = p.[0]
    let y = p.[1]
    x**2. + 3600000.0*y**4. + 100000.0*y**2.*(x - 1.)*(72.*x + 372.) + y**2. + 100000.0*(x - 1.)**2.*(6.*x + 2.9)**2.

let g191 (p:Vector<double>) =
    let x = p.[0]
    let y = p.[1]
    let seqg = [2.*x + 7200000.0*y**2.*(x - 1.) + 100000.0*y**2.*(72.*x + 372.) + 100000.0*(x - 1.)**2.*(72.*x + 348.) + (6.*x + 29.)**2.*(200000.0*x - 200000.0); 
                14400000.0*y**3. + 200000.0*y*(x - 1.)*(72.*x + 372.) + 2.*y]
    (CreateVector.DenseOfEnumerable seqg)

let h191 (p:Vector<double>) =
    let x = p.[0]
    let y = p.[1]
    let seqh = array2D [[14400000.0*y**2. + 7200000.0*(x - 1.)**2. + 200000.0*(6.*x + 29.)**2. + 2.*(72.*x + 348.)*(200000.0*x - 200000.0) + 2.;
                 14400000.0*y*(x - 1.) + 200000.0*y*(72.*x + 372.)];
                [14400000.0*y*(x - 1.) + 200000.0*y*(72.*x + 372.);
                 43200000.0*y**2. + (72.*x + 372.)*(200000.0*x - 200000.0) + 2.]]

    (CreateMatrix.DenseOfArray seqh)

[<EntryPoint>]
let main argv =
    let result = newtonL2Eq f191 g191 h191 (CreateVector.DenseOfEnumerable [|0.; 0.|])
    printfn "Optimal Point: %A" result
    printfn "Optimal Value: %A" (f191 result)
    0 // return an integer exit code