import json
import os
import re
from collections import Counter

def get_focus_info(focus_str):
    """
    Parses focus string to get category and ID.
    Example: "diver(3)" -> ("diver", "3")
    """
    if not focus_str or focus_str == "Explore":
        return None, None
    
    match = re.match(r"([a-z0-9_]+)\((\d+)\)", focus_str)
    if match:
        return match.group(1), match.group(2)
    return None, None

def abstract_state(state_atoms, focus_cat, focus_id):
    """
    Filters atoms in the state based on the focus object.
    Rules:
    - Keep atoms involving the focus object (e.g., "close_by_diver(obj0,obj3)" if focus is diver(3)).
    - Keep global atoms (player-centric, e.g., "facing_right(obj0)").
    - Discard atoms involving OTHER objects of the SAME category as focus, UNLESS they also involve the focus object.
    
    Args:
        state_atoms: List of atom strings.
        focus_cat: Category of the focus object (e.g., "diver").
        focus_id: ID of the focus object (e.g., "3").
    """
    if not focus_id:
        return state_atoms # no filtering if no specific focus
        
    if focus_cat == "surface":
        return AbstractedState(state_atoms)

    abstracted = []
    
    # We need to know which objects are of the focus category but NOT the focus ID
    # However, the atoms themselves don't carry category info directly, just IDs (objN).
    # But we know the focus ID.
    
    # Heuristic for now: 
    # 1. If atom contains "obj{focus_id}", KEEP IT.
    # 2. If atom contains "obj0" (Player), KEEP IT (mostly).
    # 3. If atom contains "obj{other_id}"... 
    #    - If predicate name suggests it's about the focus category (e.g. "visible_diver"), 
    #      AND it doesn't contain focus_id, DISCARD IT.
    
    target_token = f"obj{focus_id}"
    
    for atom in state_atoms:
        if target_token in atom:
            # Replace specific ID with generic placeholder to allow aggregation
            atom = atom.replace(target_token, "obj(focus)")
            abstracted.append(atom)
            continue
            
        # 2. Check for "same category but different object"
        # We need to guess the category from the predicate name
        # e.g., "visible_diver(obj4)" when focus is diver(3).
        
        # Predicates usually end with the category name: _diver, _shark, _enemy, etc.
        # Or start with it: diver_, shark_, etc.
        
        is_relevant_cat = False
        
        # Map focus_cat to likely predicate suffixes
        # based on NSFR predicates
        cat_map = {
            "diver": ["diver"],
            "shark": ["shark", "enemy"],
            "submarine": ["submarine", "enemy"],
            "enemymissile": ["missile"],
            "collecteddiver": ["collected_diver"]
        }
        
        likely_terms = cat_map.get(focus_cat, [focus_cat])
        
        # Check if atom predicate name contains any of the likely terms
        predicate = atom.split("(")[0]
        
        for term in likely_terms:
            if term in predicate:
                is_relevant_cat = True
                break
        
        if is_relevant_cat:
            # It's about this category, but didn't match focus_id (checked in step 1).
            # So it must be about ANOTHER object of this category.
            # DISCARD.
            continue
            
        # 3. Keep everything else (Player status, global context, other categories)
        abstracted.append(atom)
        
    return AbstractedState(abstracted)

def AbstractedState(atoms):
    return sorted(list(set(atoms)))

def compute_diff(state_a, state_b):
    """
    Returns atoms in A that are NOT in B.
    """
    set_a = set(state_a)
    set_b = set(state_b)
    return list(set_a - set_b)

    # Save Skill Library
    with open(output_skills_file, 'w') as f:
        json.dump(skills_library, f, indent=2)
    print(f"\nSaved mined skills to {output_skills_file}")

    # Generate PDDL Domain
    pddl_lines = []
    pddl_lines.append("(define (domain mined_seaquest)")
    pddl_lines.append("  (:requirements :strips :typing)")
    pddl_lines.append("  (:predicates")
    
    # Collect all predicates (for definition)
    all_predicates = set()
    
    actions_pddl = []
    
    for cid, skill in skills_library.items():
        if skill['success_rate'] == 0:
            continue
            
        action_name = skill['label'].replace(" ", "_").lower()
        
        # Preconditions
        # 1. Triggers (> 25%)
        # 2. Context (> 80%)
        preconds = []
        
        for atom, prob in skill['preconditions'].items():
            if prob > 0.25:
                preconds.append(atom)
                all_predicates.add(atom.split('(')[0]) # naive predicate extraction
        
        # Context is not fully saved in skill_def['preconditions'] currently in the loop above
        # We need to access the counts again or modify the loop above to save them
        # OPTION: Just use what's in skill_def['preconditions'] (which are currently Triggers > 20%)
        # User asked for "preconditions in the context AND effects".
        # In the loop above, we only saved Triggers to skill_def. We need to save Context too.
        # Let's fix the loop above first in a separate edit or assume we change it here.
        # Actually, I can't access `context_counts` here. 
        # I should simply rewrite the `skill_def` population logic in the previous loop to include context.
        pass

def main():
    input_file = 'labeled_clustered_segments.json'
    output_skills_file = 'mined_skills.json'
    output_pddl_file = 'mined_domain.pddl'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # ... (Group by cluster ID code remains same) ...
    segments = data.get("segments", [])
    clusters = {}
    for seg in segments:
        cid = str(seg.get("cluster_id"))
        if cid not in clusters:
            clusters[cid] = {
                "label": seg.get("label", "Unknown"),
                "segments": []
            }
        clusters[cid]["segments"].append(seg)
        
    print(f"Found {len(clusters)} clusters.")
    
    skills_library = {}
    pddl_actions = []
    all_predicates = set()

    for cid, cluster_data in sorted(clusters.items()):
        label = cluster_data['label']
        print(f"\nCluster {cid}: {label}")
        
        trigger_counts = Counter()
        context_counts = Counter()
        effect_counts = Counter()
        
        seg_count = 0
        active_seg_count = 0
        
        for seg in cluster_data["segments"]:
            focus_str = seg.get("focus")
            focus_cat, focus_id = get_focus_info(focus_str)
            
            if not focus_id: continue 
            
            seg_count += 1
            
            # Get states
            start_state = seg.get("start_frame_state", [])
            prev_state = seg.get("prev_frame_state", [])
            next_state = seg.get("next_frame_state", [])
            
            # Abstract states
            abs_start = abstract_state(start_state, focus_cat, focus_id)
            abs_prev = abstract_state(prev_state, focus_cat, focus_id)
            abs_next = abstract_state(next_state, focus_cat, focus_id)
            
            # Compute Diff/Context
            triggers = compute_diff(abs_start, abs_prev)
            context = abs_start
            added_effects = compute_diff(abs_next, abs_start)
            deleted_effects = compute_diff(abs_start, abs_next)
            
            for atom in triggers: trigger_counts[atom] += 1
            for atom in context: context_counts[atom] += 1
                
            if added_effects or deleted_effects:
                active_seg_count += 1
                for atom in added_effects: effect_counts[atom] += 1
                for atom in deleted_effects: effect_counts[f"not({atom})"] += 1
            
        if seg_count == 0: continue

        print(f"  Analyzed {seg_count} segments ({active_seg_count} active).")
        
        skill_def = {
            "id": cid,
            "label": label,
            "preconditions": {},
            "effects": {}
        }
        
        # PDDL Construction
        action_name = label.replace(" ", "_").lower()
        precond_list = []
        effect_list = []

        print("  [Preconditions/Triggers (>25%)]")
        for atom, count in trigger_counts.items():
             if atom == ".(__T__)": continue
             pct = (count / seg_count) * 100
             if pct > 25:
                 print(f"    - {atom}: {pct:.1f}%")
                 skill_def["preconditions"][atom] = pct / 100.0
                 
                 # PDDL Formatting
                 pddl_atom = atom.replace("obj(focus)", "?focus")
                 precond_list.append(pddl_atom)
                 
                 # Predicate Collection (keep abstract for definition)
                 pred_name = atom.split('(')[0]
                 all_predicates.add(pred_name)

        print("  [Context (>80%)]")
        for atom, count in context_counts.items():
             if atom == ".(__T__)": continue
             pct = (count / seg_count) * 100
             if pct > 80:
                # skill_def["preconditions"][atom] = pct / 100.0 
                pddl_atom = atom.replace("obj(focus)", "?focus")
                
                # Avoid duplicates if context overlaps with trigger (though unlikely given thresholds)
                if pddl_atom not in precond_list:
                    precond_list.append(pddl_atom)
                    pred_name = atom.split('(')[0]
                    all_predicates.add(pred_name)
                

        print("  [Effects (>25%)]")
        for atom_key, count in effect_counts.items():
             if ".(__T__)" in atom_key: continue
             
             pct_active = (count / active_seg_count) * 100 if active_seg_count > 0 else 0
             if pct_active > 25:
                 print(f"    - {atom_key}: {pct_active:.1f}%")
                 skill_def["effects"][atom_key] = pct_active / 100.0
                 
                 # PDDL Formatting
                 pddl_atom = atom_key.replace("obj(focus)", "?focus")
                 effect_list.append(pddl_atom)
                 
                 # Predicate extraction (handle 'not(atom)')
                 clean_atom = atom_key.replace("not(", "").replace(")", "")
                 pred_name = clean_atom.split('(')[0]
                 all_predicates.add(pred_name)
                 
        skills_library[cid] = skill_def
        
        pddl_act = f"  (:action {action_name}\n"
        # We assume obj0 is the player and universal, ?focus is the parameter
        # If the skill doesn't actually use ?focus (e.g. surfacing?), we might not need it, 
        # but most skills in this library are object-centric.
        # Check if ?focus is used in preconds or effects? 
        # Actually, let's just add it if the cluster logic used a focus_id (which we filtered for earlier).
        pddl_act += f"    :parameters (?focus)\n" 
        
        if precond_list:
            pddl_act += f"    :precondition (and {' '.join(precond_list)})\n"
        else:
            pddl_act += f"    :precondition (and )\n"
            
        pddl_act += f"    :effect (and\n"
        for eff in effect_list:
            pddl_act += f"      {eff}\n"
        pddl_act += f"    )\n  )"
        pddl_actions.append(pddl_act)

    # Save Skill Library
    with open(output_skills_file, 'w') as f:
        json.dump(skills_library, f, indent=2)
    print(f"\nSaved mined skills to {output_skills_file}")
    
    # Save PDDL
    with open(output_pddl_file, 'w') as f:
        f.write("(define (domain local_mined)\n")
        f.write("  (:requirements :strips)\n")
        f.write("  (:predicates\n")
        for p in sorted(list(all_predicates)):
            f.write(f"    ({p} ?x ?y)\n") # Dummy signature
        f.write("  )\n\n")
        
        for act in pddl_actions:
            f.write(act + "\n\n")
            
        f.write(")\n")
    print(f"Saved PDDL domain to {output_pddl_file}")

if __name__ == "__main__":
    main()
