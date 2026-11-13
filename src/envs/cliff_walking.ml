(* CliffWalking, same dynamics as Gymnasium's CliffWalking-v0 (Sutton & Barto,
   example 6.6).

   A 4x12 grid. The agent starts in the bottom-left corner and must reach the
   bottom-right corner. Every step costs 1. Stepping onto the cliff between
   them costs 100 and sends the agent back to the start. *)

let rows = 4
let cols = 12
let start = (rows - 1) * cols
let goal = (rows * cols) - 1

(* actions: 0 up, 1 right, 2 down, 3 left *)
let moves = [| -1, 0; 0, 1; 1, 0; 0, -1 |]

let arrows = [| "^"; ">"; "v"; "<" |]

let is_cliff s = s / cols = rows - 1 && s mod cols > 0 && s mod cols < cols - 1

let make () : Mdp.t =
  let move s a =
    let dr, dc = moves.(a) in
    let r = max 0 (min (rows - 1) ((s / cols) + dr))
    and c = max 0 (min (cols - 1) ((s mod cols) + dc)) in
    (r * cols) + c
  in
  let transitions =
    Array.init (rows * cols) (fun s ->
        Array.init 4 (fun a ->
            let next = move s a in
            if is_cliff next
            then [ { Mdp.prob = 1.; next = start; reward = -100.; terminal = false } ]
            else [ { Mdp.prob = 1.; next; reward = -1.; terminal = next = goal } ]))
  in
  let initial = Array.init (rows * cols) (fun s -> if s = start then 1. else 0.) in
  { n_states = rows * cols; n_actions = 4; transitions; initial }


let print_policy policy =
  for r = 0 to rows - 1 do
    for c = 0 to cols - 1 do
      let s = (r * cols) + c in
      print_string (if is_cliff s then " C" else if s = goal then " G" else " " ^ arrows.(policy.(s)))
    done;
    print_newline ()
  done
