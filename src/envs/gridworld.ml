(* Deterministic grid worlds described by a text map.

   'S' start, 'G' goal (terminal), '#' wall, anything else is a free cell.
   States are numbered row by row. Moving into a wall or off the grid leaves
   the agent in place. Without any 'S', episodes start in a random free cell. *)

type t =
  { rows : int
  ; cols : int
  ; cells : string array
  ; mdp : Mdp.t
  }

(* actions: 0 up, 1 right, 2 down, 3 left *)
let moves = [| -1, 0; 0, 1; 1, 0; 0, -1 |]

let arrows = [| "^"; ">"; "v"; "<" |]

(* every move costs [step_reward]; entering a goal adds [goal_reward] *)
let make ?(step_reward = -1.) ?(goal_reward = 0.) cells =
  let rows = Array.length cells
  and cols = String.length cells.(0) in
  let cell s = cells.(s / cols).[s mod cols] in
  let absorbing s = cell s = 'G' || cell s = '#' in
  let move s a =
    let dr, dc = moves.(a) in
    let r = (s / cols) + dr
    and c = (s mod cols) + dc in
    if r < 0 || r >= rows || c < 0 || c >= cols || cells.(r).[c] = '#' then s else (r * cols) + c
  in
  let transitions =
    Array.init (rows * cols) (fun s ->
        Array.init 4 (fun a ->
            if absorbing s
            then [ { Mdp.prob = 1.; next = s; reward = 0.; terminal = true } ]
            else (
              let next = move s a in
              let goal = cell next = 'G' in
              let reward = step_reward +. if goal then goal_reward else 0. in
              [ { Mdp.prob = 1.; next; reward; terminal = goal } ])))
  in
  let states = List.init (rows * cols) Fun.id in
  let starts =
    match List.filter (fun s -> cell s = 'S') states with
    | [] -> List.filter (fun s -> not (absorbing s)) states
    | starts -> starts
  in
  let p = 1. /. float_of_int (List.length starts) in
  let initial = Array.init (rows * cols) (fun s -> if List.mem s starts then p else 0.) in
  { rows; cols; cells; mdp = { n_states = rows * cols; n_actions = 4; transitions; initial } }


(* the 4x4 grid of Sutton & Barto, example 4.1 *)
let sutton_4x4 () = make [| "G..."; "...."; "...."; "...G" |]

let render g show =
  for r = 0 to g.rows - 1 do
    for c = 0 to g.cols - 1 do
      let s = (r * g.cols) + c in
      match g.cells.(r).[c] with
      | '#' -> Printf.printf "%7s" "#"
      | 'G' -> Printf.printf "%7s" "G"
      | _ -> Printf.printf "%7s" (show s)
    done;
    print_newline ()
  done


let print_values g v = render g (fun s -> Printf.sprintf "%.1f" v.(s))

let print_policy g policy = render g (fun s -> arrows.(policy.(s)))
