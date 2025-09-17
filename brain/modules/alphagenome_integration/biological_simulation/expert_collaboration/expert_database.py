#!/usr/bin/env python3
"""Expert Collaboration Database - Manages relationships with developmental biology experts.

This module facilitates collaboration with researchers specializing in morphogen signaling,
embryonic development, and neural tube formation for scientific validation of QUARK's
morphogen solver parameters.

Integration: Core expert validation system for BiologicalSimulator scientific accuracy.
Rationale: Ensures biological fidelity through direct expert collaboration and validation.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ExpertiseArea(Enum):
    """Areas of developmental biology expertise relevant to morphogen modeling."""
    MORPHOGEN_SIGNALING = "morphogen_signaling"
    NEURAL_TUBE_DEVELOPMENT = "neural_tube_development" 
    SHH_SIGNALING = "shh_signaling"
    BMP_SIGNALING = "bmp_signaling"
    WNT_SIGNALING = "wnt_signaling"
    FGF_SIGNALING = "fgf_signaling"
    EMBRYONIC_PATTERNING = "embryonic_patterning"
    DEVELOPMENTAL_BIOLOGY = "developmental_biology"
    SYSTEMS_BIOLOGY = "systems_biology"
    MATHEMATICAL_MODELING = "mathematical_modeling"

class CollaborationStatus(Enum):
    """Status of collaboration with expert."""
    POTENTIAL = "potential"
    CONTACTED = "contacted"
    RESPONSIVE = "responsive"
    COLLABORATING = "collaborating"
    VALIDATED = "validated"
    ONGOING = "ongoing"

@dataclass
class Expert:
    """Represents a developmental biology expert for collaboration."""
    name: str
    email: str
    institution: str
    department: str
    expertise_areas: List[ExpertiseArea]
    h_index: Optional[int] = None
    orcid_id: Optional[str] = None
    key_publications: List[str] = None
    collaboration_status: CollaborationStatus = CollaborationStatus.POTENTIAL
    contact_date: Optional[str] = None
    response_date: Optional[str] = None
    notes: str = ""
    validated_parameters: List[str] = None
    
    def __post_init__(self):
        if self.key_publications is None:
            self.key_publications = []
        if self.validated_parameters is None:
            self.validated_parameters = []

@dataclass
class ValidationSession:
    """Represents an expert validation session for parameters or results."""
    session_id: str
    expert_name: str
    date: str
    session_type: str  # "parameter_review", "result_validation", "workshop"
    topics_covered: List[str]
    validation_outcomes: Dict[str, Any]
    expert_feedback: str
    follow_up_actions: List[str]
    files_reviewed: List[str] = None
    
    def __post_init__(self):
        if self.files_reviewed is None:
            self.files_reviewed = []

class ExpertDatabase:
    """Manages expert collaboration database and validation tracking."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize expert database."""
        if db_path is None:
            db_path = Path(__file__).parent / "expert_collaboration.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    institution TEXT NOT NULL,
                    department TEXT NOT NULL,
                    expertise_areas TEXT NOT NULL,
                    h_index INTEGER,
                    orcid_id TEXT,
                    key_publications TEXT,
                    collaboration_status TEXT NOT NULL,
                    contact_date TEXT,
                    response_date TEXT,
                    notes TEXT,
                    validated_parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Validation sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    expert_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    topics_covered TEXT NOT NULL,
                    validation_outcomes TEXT NOT NULL,
                    expert_feedback TEXT,
                    follow_up_actions TEXT,
                    files_reviewed TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_expert(self, expert: Expert) -> bool:
        """Add new expert to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO experts (
                        name, email, institution, department, expertise_areas,
                        h_index, orcid_id, key_publications, collaboration_status,
                        contact_date, response_date, notes, validated_parameters
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    expert.name,
                    expert.email, 
                    expert.institution,
                    expert.department,
                    json.dumps([area.value for area in expert.expertise_areas]),
                    expert.h_index,
                    expert.orcid_id,
                    json.dumps(expert.key_publications),
                    expert.collaboration_status.value,
                    expert.contact_date,
                    expert.response_date,
                    expert.notes,
                    json.dumps(expert.validated_parameters)
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # Email already exists
    
    def update_expert_status(self, email: str, status: CollaborationStatus, 
                           notes: str = "") -> bool:
        """Update expert collaboration status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experts 
                SET collaboration_status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE email = ?
            """, (status.value, notes, email))
            conn.commit()
            return cursor.rowcount > 0
    
    def record_validation_session(self, session: ValidationSession) -> bool:
        """Record expert validation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO validation_sessions (
                        session_id, expert_name, date, session_type, topics_covered,
                        validation_outcomes, expert_feedback, follow_up_actions,
                        files_reviewed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.expert_name,
                    session.date,
                    session.session_type,
                    json.dumps(session.topics_covered),
                    json.dumps(session.validation_outcomes),
                    session.expert_feedback,
                    json.dumps(session.follow_up_actions),
                    json.dumps(session.files_reviewed)
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # Session ID already exists
    
    def get_experts_by_expertise(self, expertise: ExpertiseArea) -> List[Expert]:
        """Get experts with specific expertise area."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experts")
            experts = []
            
            for row in cursor.fetchall():
                expertise_areas = [ExpertiseArea(area) for area in json.loads(row[5])]
                if expertise in expertise_areas:
                    expert = Expert(
                        name=row[1],
                        email=row[2],
                        institution=row[3],
                        department=row[4],
                        expertise_areas=expertise_areas,
                        h_index=row[6],
                        orcid_id=row[7],
                        key_publications=json.loads(row[8]) if row[8] else [],
                        collaboration_status=CollaborationStatus(row[9]),
                        contact_date=row[10],
                        response_date=row[11],
                        notes=row[12],
                        validated_parameters=json.loads(row[13]) if row[13] else []
                    )
                    experts.append(expert)
            
            return experts
    
    def get_validation_history(self, expert_name: str) -> List[ValidationSession]:
        """Get validation session history for expert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_sessions WHERE expert_name = ?
                ORDER BY date DESC
            """, (expert_name,))
            
            sessions = []
            for row in cursor.fetchall():
                session = ValidationSession(
                    session_id=row[1],
                    expert_name=row[2],
                    date=row[3],
                    session_type=row[4],
                    topics_covered=json.loads(row[5]),
                    validation_outcomes=json.loads(row[6]),
                    expert_feedback=row[7],
                    follow_up_actions=json.loads(row[8]),
                    files_reviewed=json.loads(row[9]) if row[9] else []
                )
                sessions.append(session)
            
            return sessions
    
    def generate_collaboration_report(self) -> Dict[str, Any]:
        """Generate comprehensive collaboration status report."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Expert statistics
            cursor.execute("SELECT collaboration_status, COUNT(*) FROM experts GROUP BY collaboration_status")
            status_counts = dict(cursor.fetchall())
            
            # Expertise area coverage
            cursor.execute("SELECT expertise_areas FROM experts")
            all_expertise = []
            for row in cursor.fetchall():
                all_expertise.extend(json.loads(row[0]))
            
            from collections import Counter
            expertise_coverage = Counter(all_expertise)
            
            # Recent validation sessions
            cursor.execute("""
                SELECT COUNT(*) FROM validation_sessions 
                WHERE date >= date('now', '-30 days')
            """)
            recent_sessions = cursor.fetchone()[0]
            
            return {
                "expert_count": sum(status_counts.values()),
                "collaboration_status": status_counts,
                "expertise_coverage": dict(expertise_coverage),
                "recent_validation_sessions": recent_sessions,
                "generated_at": datetime.now().isoformat()
            }

def initialize_expert_database_with_targets():
    """Initialize database with target experts for morphogen research."""
    db = ExpertDatabase()
    
    # Target experts (hypothetical - would be real researchers)
    target_experts = [
        Expert(
            name="Dr. Sarah Chen",
            email="s.chen@university.edu",
            institution="Stanford University",
            department="Developmental Biology",
            expertise_areas=[ExpertiseArea.SHH_SIGNALING, ExpertiseArea.NEURAL_TUBE_DEVELOPMENT],
            h_index=45,
            key_publications=[
                "Sonic hedgehog gradient formation in neural tube (Nature, 2023)",
                "Mathematical modeling of morphogen diffusion (Cell, 2022)"
            ]
        ),
        Expert(
            name="Dr. Michael Rodriguez",
            email="m.rodriguez@institute.org", 
            institution="MIT",
            department="Systems Biology",
            expertise_areas=[ExpertiseArea.BMP_SIGNALING, ExpertiseArea.MATHEMATICAL_MODELING],
            h_index=38,
            key_publications=[
                "BMP gradient dynamics in embryonic development (Science, 2023)",
                "Quantitative analysis of morphogen networks (Dev Cell, 2022)"
            ]
        ),
        Expert(
            name="Dr. Emma Thompson",
            email="e.thompson@lab.ac.uk",
            institution="Cambridge University", 
            department="Embryology",
            expertise_areas=[ExpertiseArea.WNT_SIGNALING, ExpertiseArea.FGF_SIGNALING],
            h_index=52,
            key_publications=[
                "Wnt/FGF crosstalk in neural patterning (Development, 2023)",
                "Temporal dynamics of growth factor gradients (PNAS, 2022)"
            ]
        )
    ]
    
    for expert in target_experts:
        db.add_expert(expert)
    
    print("✅ Expert database initialized with target collaborators")
    return db

if __name__ == "__main__":
    # Initialize expert collaboration system
    db = initialize_expert_database_with_targets()
    
    # Generate initial report
    report = db.generate_collaboration_report()
    print("\n📊 Expert Collaboration Status:")
    print(f"   Total experts: {report['expert_count']}")
    print(f"   Status distribution: {report['collaboration_status']}")
    print(f"   Expertise coverage: {report['expertise_coverage']}")