from collections import defaultdict


def reformat_pdb(input_pdb, output_pdb):
    """
    Reformat PDB file with custom atom naming (ABCC format).
    - A: Element type (1st character)
    - B: Residue number (2nd character)
    - CC: Element counter (1 or 2 digit, starting from 1 for each residue)
    """
    with open(input_pdb, "r") as f_in, open(output_pdb, "w") as f_out:
        element_counters = defaultdict(lambda: defaultdict(int))
        current_residue = None

        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                # Parse fields (assuming fixed-width PDB format)
                atom_name = line[12:16].strip()
                residue_number = line[22:26].strip()
                element = (
                    line[76:78].strip() or atom_name[0]
                )  # Fallback to 1st char if no element

                # Reset counters for new residue
                if residue_number != current_residue:
                    element_counters[residue_number].clear()
                    current_residue = residue_number

                # Update counter for this element in residue
                element_counters[residue_number][element] += 1
                element_counter = element_counters[residue_number][element]

                # Generate new atom name (ABCC format)
                new_atom_name = f"{element[0]}{residue_number[-1]}{element_counter:d}"
                new_atom_name = new_atom_name.ljust(4)  # Pad to 4 chars

                # Reconstruct line
                new_line = (
                    f"HETATM{line[6:11]}  {new_atom_name}{'UNL'.ljust(3)} "
                    f"{line[21:22]}{'1'.rjust(4)}{line[26:76]}"
                    f"{element.rjust(2)}{line[78:]}"
                )
                f_out.write(new_line)
            else:
                f_out.write(line)  # Write non-ATOM lines unchanged


if __name__ == "__main__":
    from pathlib import Path

    input_dir = Path("../../../Data/Hexene")
    output_dir = Path("../../../Data/For_JG/Hexene")
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in input_dir.glob("*.pdb"):
        if "2015_Wang" not in f.name:
            continue

        output_path = output_dir / f.name
        reformat_pdb(f, output_path)
        print(f"Reformatted PDB {f.name} -> {output_path}")
