(* A Python child process that answers each request with one line.

   Python runs in its own process, not inside the OCaml one, so the native
   libraries it loads (Conda's BLAS, SDL, ...) never mix with Owl's: Conda's
   numerical libraries loaded into an Owl process were seen to change its
   results. Floats cross the pipe in
   hexadecimal ("%h" / float.hex), so values arrive bit-exact.

   The interpreter is $OWL_RLLIB_PYTHON, or python3. It runs with -I, so the
   current directory and PYTHON* variables cannot change what is imported.
   The child exits when this program does: its stdin then reaches EOF. *)

type t = { name : string; input : in_channel; output : out_channel }

(* Library output that would corrupt the protocol goes to stderr instead;
   the protocol uses a private copy of the original stdout. *)
let prelude =
  {|
import os, sys, traceback
_out = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)

def reply(*tokens):
    _out.write(' '.join(map(str, tokens)) + '\n')
    _out.flush()

def hexes(xs):
    return ' '.join(float(x).hex() for x in xs)

def fail(e):
    traceback.print_exc()
    reply('error', ' '.join(repr(e).split()))

def serve(handle):
    for line in sys.stdin:
        words = line.split()
        try:
            handle(words[0], words[1:])
        except Exception as e:
            fail(e)
|}

let receive t =
  match input_line t.input with
  | line -> (
      match String.split_on_char ' ' line with
      | "error" :: message -> failwith (t.name ^ ": " ^ String.concat " " message)
      | words -> words)
  | exception End_of_file -> failwith (t.name ^ ": the Python process exited (its error is on stderr)")

(* Starts [script] with [args] in sys.argv[1:] and returns its first reply. *)
let start ~name ~script args =
  let python = Option.value (Sys.getenv_opt "OWL_RLLIB_PYTHON") ~default:"python3" in
  let input, output = Unix.open_process_args python (Array.append [| python; "-I"; "-c"; prelude ^ script |] args) in
  let t = { name; input; output } in
  t, receive t

let request t line =
  output_string t.output line;
  output_char t.output '\n';
  flush t.output;
  receive t

let hex x = Printf.sprintf "%h" x
