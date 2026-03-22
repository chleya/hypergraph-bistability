"""
Import MiniMax Skills into Unified Node System
=============================================

This script parses MiniMax AI skills and imports them into the 
unified node system for use by the agent.
"""

import os
import sys
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, "F:/hypergraph_bistability/src")

from hypergraph_bistability.memory.unified_node import (
    UnifiedNodeManager,
    NodeType,
    SkillDefinition,
)


def parse_skill_md(skill_path: str) -> dict:
    """Parse a SKILL.md file and extract metadata."""
    with open(skill_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract YAML frontmatter
    frontmatter = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_text = parts[1]
            # Simple YAML parsing
            for line in yaml_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    value = value.strip().strip('"')
                    frontmatter[key.strip()] = value
    
    # Extract name from content
    name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = name_match.group(1).strip() if name_match else ""
    
    # Extract description from frontmatter or first paragraph
    description = frontmatter.get('description', '')
    if not description:
        desc_match = re.search(r'^.+?$', content.split('\n\n', 1)[1] if '\n\n' in content else content)
        description = desc_match.group(0).strip()[:200] if desc_match else ""
    
    return {
        'name': frontmatter.get('name', Path(skill_path).parent.name),
        'description': description,
        'category': frontmatter.get('metadata', {}).get('category', 'general') if isinstance(frontmatter.get('metadata'), dict) else 'general',
        'license': frontmatter.get('license', 'MIT'),
        'content': content,
    }


def import_skill(manager: UnifiedNodeManager, skill_dir: str) -> int:
    """Import a single skill into the unified node system."""
    skill_path = os.path.join(skill_dir, 'SKILL.md')
    if not os.path.exists(skill_path):
        print(f"  [SKIP] No SKILL.md found")
        return 0
    
    # Parse skill
    skill_data = parse_skill_md(skill_path)
    
    # Create skill definition
    skill_def = SkillDefinition(
        name=skill_data['name'],
        code=f"# Imported from MiniMax AI Skills\n# Category: {skill_data['category']}\n\n# See original SKILL.md for full documentation",
        description=skill_data['description'],
        category=skill_data['category'],
        parameters={
            'license': skill_data['license'],
            'imported_from': 'MiniMax-AI/skills',
            'source_path': skill_dir,
        }
    )
    
    # Add skill to manager
    skill_id = manager.store.add_node(
        content=skill_data['name'],
        node_type=NodeType.SKILL,
        skill_def=skill_def,
        effectiveness=0.8,  # High importance for imported skills
    )
    
    # Also add the full content as a memory node for reference
    mem_id = manager.store.add_node(
        content=f"# {skill_data['name']}\n\n{skill_data['content'][:5000]}",  # Truncate if too long
        node_type=NodeType.MEMORY,
        effectiveness=0.5,
        metadata={
            'imported_skill': skill_data['name'],
            'type': 'skill_documentation',
        }
    )
    
    return skill_id


def import_all_skills(skills_dir: str, db_path: str = "minimax_skills.db") -> dict:
    """Import all MiniMax skills into the unified node system."""
    manager = UnifiedNodeManager(db_path)
    
    results = {
        'imported': [],
        'skipped': [],
        'errors': [],
    }
    
    # Get all skill directories
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        print(f"Error: Skills directory not found: {skills_dir}")
        return results
    
    skill_dirs = [d for d in skills_path.iterdir() if d.is_dir()]
    print(f"Found {len(skill_dirs)} skills to import\n")
    
    for skill_dir in sorted(skill_dirs):
        skill_name = skill_dir.name
        print(f"Importing: {skill_name}...", end=" ")
        
        try:
            skill_id = import_skill(manager, str(skill_dir))
            if skill_id > 0:
                print(f"OK (id={skill_id})")
                results['imported'].append(skill_name)
            else:
                print(f"SKIP")
                results['skipped'].append(skill_name)
        except Exception as e:
            print(f"ERROR: {e}")
            results['errors'].append({'skill': skill_name, 'error': str(e)})
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Import Summary")
    print(f"{'='*50}")
    print(f"Imported: {len(results['imported'])}")
    print(f"Skipped:  {len(results['skipped'])}")
    print(f"Errors:   {len(results['errors'])}")
    
    # Print stats
    stats = manager.get_stats()
    print(f"\nTotal nodes in database: {stats['total']}")
    print(f"By type: {stats['by_type']}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Import MiniMax AI Skills')
    parser.add_argument('--skills-dir', default='F:/hypergraph_bistability/temp_skills/skills',
                        help='Path to skills directory')
    parser.add_argument('--db', default='minimax_skills.db',
                        help='Output database path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MiniMax AI Skills Importer")
    print("="*60)
    print(f"Skills directory: {args.skills_dir}")
    print(f"Output database:   {args.db}")
    print()
    
    results = import_all_skills(args.skills_dir, args.db)
    
    return 0 if not results['errors'] else 1


if __name__ == "__main__":
    sys.exit(main())
