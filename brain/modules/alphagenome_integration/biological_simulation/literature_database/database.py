#!/usr/bin/env python3
"""Literature Database Core - SQLite database operations for morphogen parameter storage.

This module handles all database operations for storing and retrieving morphogen parameters,
citations, and experimental data from developmental biology literature.

Integration: Core storage layer for literature_database package.
Rationale: Centralized database operations with clean separation from business logic.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .types import (
    Parameter, Citation, ParameterSet, MorphogenType, ParameterType, 
    DevelopmentalStage, ConfidenceLevel
)

class ParameterDatabase:
    """Manages SQLite database operations for morphogen parameters."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize parameter database."""
        if db_path is None:
            db_path = Path(__file__).parent / "morphogen_parameters.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Parameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_id TEXT UNIQUE NOT NULL,
                    morphogen TEXT NOT NULL,
                    parameter_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    std_deviation REAL,
                    min_value REAL,
                    max_value REAL,
                    developmental_stage TEXT,
                    species TEXT NOT NULL,
                    experimental_method TEXT,
                    tissue_type TEXT,
                    confidence_level TEXT NOT NULL,
                    notes TEXT,
                    date_added TEXT NOT NULL,
                    expert_validated INTEGER DEFAULT 0,
                    expert_comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Citations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    citation_id TEXT UNIQUE NOT NULL,
                    authors TEXT NOT NULL,
                    title TEXT NOT NULL,
                    journal TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    volume TEXT,
                    pages TEXT,
                    doi TEXT,
                    pmid TEXT,
                    abstract TEXT,
                    experimental_methods TEXT,
                    species_studied TEXT,
                    relevance_score REAL DEFAULT 0.0,
                    quality_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Parameter-Citation relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_citations (
                    parameter_id TEXT NOT NULL,
                    citation_id TEXT NOT NULL,
                    PRIMARY KEY (parameter_id, citation_id),
                    FOREIGN KEY (parameter_id) REFERENCES parameters(parameter_id),
                    FOREIGN KEY (citation_id) REFERENCES citations(citation_id)
                )
            """)
            
            # Parameter sets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_id TEXT UNIQUE NOT NULL,
                    morphogen TEXT NOT NULL,
                    developmental_stage TEXT NOT NULL,
                    experimental_conditions TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    citation_id TEXT NOT NULL,
                    confidence_assessment TEXT,
                    expert_review_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_parameter(self, parameter: Parameter, citation_ids: List[str] = None) -> bool:
        """Add parameter with optional citations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert parameter
                cursor.execute("""
                    INSERT INTO parameters (
                        parameter_id, morphogen, parameter_type, value, unit,
                        std_deviation, min_value, max_value, developmental_stage,
                        species, experimental_method, tissue_type, confidence_level,
                        notes, date_added, expert_validated, expert_comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    parameter.parameter_id,
                    parameter.morphogen.value,
                    parameter.parameter_type.value,
                    parameter.value,
                    parameter.unit,
                    parameter.std_deviation,
                    parameter.min_value,
                    parameter.max_value,
                    parameter.developmental_stage.value if parameter.developmental_stage else None,
                    parameter.species,
                    parameter.experimental_method,
                    parameter.tissue_type,
                    parameter.confidence_level.value,
                    parameter.notes,
                    parameter.date_added,
                    1 if parameter.expert_validated else 0,
                    parameter.expert_comments
                ))
                
                # Link citations if provided
                if citation_ids:
                    for citation_id in citation_ids:
                        cursor.execute("""
                            INSERT OR IGNORE INTO parameter_citations (parameter_id, citation_id)
                            VALUES (?, ?)
                        """, (parameter.parameter_id, citation_id))
                
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def add_citation(self, citation: Citation) -> bool:
        """Add literature citation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO citations (
                        citation_id, authors, title, journal, year, volume, pages,
                        doi, pmid, abstract, experimental_methods, species_studied,
                        relevance_score, quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    citation.citation_id,
                    citation.authors,
                    citation.title,
                    citation.journal,
                    citation.year,
                    citation.volume,
                    citation.pages,
                    citation.doi,
                    citation.pmid,
                    citation.abstract,
                    citation.experimental_methods,
                    citation.species_studied,
                    citation.relevance_score,
                    citation.quality_score
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_parameters_by_morphogen(self, morphogen: MorphogenType, 
                                   parameter_type: Optional[ParameterType] = None) -> List[Parameter]:
        """Get parameters for specific morphogen and optionally parameter type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM parameters WHERE morphogen = ?"
            params = [morphogen.value]
            
            if parameter_type:
                query += " AND parameter_type = ?"
                params.append(parameter_type.value)
            
            cursor.execute(query, params)
            
            parameters = []
            for row in cursor.fetchall():
                parameter = Parameter(
                    parameter_id=row[1],
                    morphogen=MorphogenType(row[2]),
                    parameter_type=ParameterType(row[3]),
                    value=row[4],
                    unit=row[5],
                    std_deviation=row[6],
                    min_value=row[7],
                    max_value=row[8],
                    developmental_stage=DevelopmentalStage(row[9]) if row[9] else None,
                    species=row[10],
                    experimental_method=row[11],
                    tissue_type=row[12],
                    confidence_level=ConfidenceLevel(row[13]),
                    notes=row[14],
                    date_added=row[15],
                    expert_validated=bool(row[16]),
                    expert_comments=row[17]
                )
                parameters.append(parameter)
            
            return parameters
    
    def search_citations_by_keyword(self, keywords: List[str]) -> List[Citation]:
        """Search citations by keywords in title or abstract."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build search query
            conditions = []
            params = []
            for keyword in keywords:
                conditions.append("(title LIKE ? OR abstract LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            query = f"SELECT * FROM citations WHERE {' AND '.join(conditions)}"
            cursor.execute(query, params)
            
            citations = []
            for row in cursor.fetchall():
                citation = Citation(
                    citation_id=row[1],
                    authors=row[2],
                    title=row[3],
                    journal=row[4],
                    year=row[5],
                    volume=row[6],
                    pages=row[7],
                    doi=row[8],
                    pmid=row[9],
                    abstract=row[10],
                    experimental_methods=row[11],
                    species_studied=row[12],
                    relevance_score=row[13],
                    quality_score=row[14]
                )
                citations.append(citation)
            
            return citations
    
    def update_expert_validation(self, parameter_id: str, validated: bool, 
                               comments: str = "") -> bool:
        """Update expert validation status for parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE parameters 
                SET expert_validated = ?, expert_comments = ?
                WHERE parameter_id = ?
            """, (1 if validated else 0, comments, parameter_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_database_statistics(self) -> Dict[str, any]:
        """Get comprehensive database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count parameters by morphogen
            cursor.execute("""
                SELECT morphogen, COUNT(*) FROM parameters GROUP BY morphogen
            """)
            morphogen_counts = dict(cursor.fetchall())
            
            # Count by parameter type
            cursor.execute("""
                SELECT parameter_type, COUNT(*) FROM parameters GROUP BY parameter_type
            """)
            parameter_type_counts = dict(cursor.fetchall())
            
            # Expert validation status
            cursor.execute("""
                SELECT expert_validated, COUNT(*) FROM parameters GROUP BY expert_validated
            """)
            validation_status = dict(cursor.fetchall())
            
            # Citation count
            cursor.execute("SELECT COUNT(*) FROM citations")
            citation_count = cursor.fetchone()[0]
            
            return {
                "total_parameters": sum(morphogen_counts.values()),
                "morphogen_distribution": morphogen_counts,
                "parameter_type_distribution": parameter_type_counts,
                "expert_validation_status": validation_status,
                "total_citations": citation_count
            }

# Export main class
__all__ = ["ParameterDatabase"]