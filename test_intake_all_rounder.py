"""
ALL-ROUNDER COMPREHENSIVE TEST SUITE FOR INTAKE FORM
====================================================

Tests ALL functionality in ALL situations:
1. Multi-symptom handling (various combinations)
2. Medication question wording (symptom-specific)
3. Travel question prevention (when checkbox not ticked)
4. Chronic monitoring/screening questions (all chronic diseases)
5. Question ordering (priority sequence)
6. Safety constraints (gender, age, travel)
7. Edge cases and combinations
8. Both English and Spanish

Run with: python3 test_intake_all_rounder.py
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clinicai.adapters.external.question_service_openai import OpenAIQuestionService

# Test results tracking
test_results = {
    "total_tests": 0,
    "passed": [],
    "failed": [],
    "warnings": [],
    "test_details": {}
}

def print_header(text: str, char: str = "="):
    print("\n" + char * 100)
    print(f"  {text}")
    print(char * 100)

def print_section(text: str):
    print(f"\n{'─' * 100}")
    print(f"  {text}")
    print(f"{'─' * 100}")

def print_test(test_name: str, test_num: int, total: int):
    print(f"\n{'=' * 100}")
    print(f"TEST {test_num}/{total}: {test_name}")
    print(f"{'=' * 100}")

def assert_test(condition: bool, message: str, test_name: str, severity: str = "error"):
    """Assert a test condition"""
    test_results["total_tests"] += 1
    if condition:
        print(f"  ✅ PASS: {message}")
        test_results["passed"].append(f"{test_name}: {message}")
        if test_name not in test_results["test_details"]:
            test_results["test_details"][test_name] = {"passed": [], "failed": [], "warnings": []}
        test_results["test_details"][test_name]["passed"].append(message)
    else:
        print(f"  ❌ FAIL: {message}")
        test_results["failed"].append(f"{test_name}: {message}")
        if test_name not in test_results["test_details"]:
            test_results["test_details"][test_name] = {"passed": [], "failed": [], "warnings": []}
        test_results["test_details"][test_name]["failed"].append(message)

def warn_test(message: str, test_name: str):
    """Add a warning"""
    print(f"  ⚠️  WARN: {message}")
    test_results["warnings"].append(f"{test_name}: {message}")
    if test_name not in test_results["test_details"]:
        test_results["test_details"][test_name] = {"passed": [], "failed": [], "warnings": []}
    test_results["test_details"][test_name]["warnings"].append(message)

async def run_test_scenario(
    test_name: str,
    chief_complaint: str,
    patient_age: int = 35,
    patient_gender: str = "male",
    recently_travelled: bool = False,
    language: str = "en",
    max_questions: int = 10,
    expected_checks: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run a test scenario and return detailed results"""
    
    if verbose:
        print(f"\n📋 Scenario: {chief_complaint}")
        print(f"   Patient: {patient_age}y/o {patient_gender}")
        print(f"   Recently travelled: {recently_travelled}")
        print(f"   Language: {language}")
    
    service = OpenAIQuestionService()
    
    asked_questions = []
    previous_answers = []
    
    # Generate first question
    try:
        first_q = await service.generate_first_question(chief_complaint, language)
        asked_questions.append(first_q)
        if verbose:
            print(f"\n   Q1: {first_q}")
        
        # Simulate first answer
        first_answer = chief_complaint
        previous_answers.append(first_answer)
        
        # Generate subsequent questions
        for i in range(1, max_questions):
            try:
                next_q = await service.generate_next_question(
                    disease=chief_complaint,
                    previous_answers=previous_answers,
                    asked_questions=asked_questions,
                    current_count=len(asked_questions),
                    max_count=max_questions,
                    recently_travelled=recently_travelled,
                    patient_gender=patient_gender,
                    patient_age=patient_age,
                    language=language
                )
                
                if next_q and next_q.strip() and next_q != "COMPLETE":
                    asked_questions.append(next_q)
                    if verbose:
                        print(f"   Q{len(asked_questions)}: {next_q}")
                    
                    # Simulate answer
                    answer = f"Answer to Q{len(asked_questions)}"
                    previous_answers.append(answer)
                else:
                    break
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error generating question {i+1}: {e}")
                break
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return {"questions": [], "error": str(e)}
    
    if verbose:
        print(f"\n   📊 Generated {len(asked_questions)} questions")
    
    # Run validation checks
    validation_results = {}
    if expected_checks:
        validation_results = run_validation_checks(
            test_name, asked_questions, expected_checks, 
            recently_travelled, chief_complaint, patient_gender, patient_age
        )
    
    return {
        "questions": asked_questions,
        "validation": validation_results,
        "count": len(asked_questions)
    }

def run_validation_checks(
    test_name: str, 
    questions: List[str], 
    checks: Dict[str, Any],
    recently_travelled: bool,
    chief_complaint: str,
    patient_gender: str,
    patient_age: int
) -> Dict[str, Any]:
    """Run comprehensive validation checks"""
    
    results = {}
    questions_lower = [q.lower() for q in questions]
    questions_text = " ".join(questions_lower)
    
    # ========================================================================
    # CHECK 1: Multi-Symptom Handling
    # ========================================================================
    if checks.get("check_multi_symptom"):
        symptoms = checks["check_multi_symptom"]
        if isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(",")]
        
        symptom_mentions = {}
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            mentions = [i+1 for i, q in enumerate(questions) if symptom_lower in q.lower()]
            symptom_mentions[symptom] = mentions
        
        all_mentioned = all(len(mentions) > 0 for mentions in symptom_mentions.values())
        
        assert_test(
            all_mentioned,
            f"All symptoms mentioned: {symptoms}. Mentions: {symptom_mentions}",
            test_name
        )
        results["multi_symptom"] = {
            "passed": all_mentioned,
            "symptoms": symptoms,
            "mentions": symptom_mentions
        }
    
    # ========================================================================
    # CHECK 2: Medication Question Wording (Symptom-Specific)
    # ========================================================================
    if checks.get("check_medication_wording"):
        medication_questions = [
            (i+1, q) for i, q in enumerate(questions) 
            if any(word in q.lower() for word in [
                "medication", "medicine", "medicamento", 
                "treatment", "tratamiento", "remedy", "remedio"
            ])
        ]
        
        if medication_questions:
            q_num, med_q = medication_questions[0]
            
            # Check if symptom-specific
            symptom_specific_indicators = [
                any(symptom.lower() in med_q.lower() for symptom in chief_complaint.lower().split()),
                "for your" in med_q.lower() or "para su" in med_q.lower(),
                "for this" in med_q.lower() or "para esto" in med_q.lower(),
                "for your" in med_q.lower() or "para tu" in med_q.lower(),
                "home remedy" in med_q.lower() or "remedio casero" in med_q.lower(),
                "tried for" in med_q.lower() or "probado para" in med_q.lower()
            ]
            
            is_symptom_specific = any(symptom_specific_indicators)
            
            # Check if generic (bad)
            generic_indicators = [
                "what medications are you currently taking" in med_q.lower(),
                "qué medicamentos está tomando actualmente" in med_q.lower(),
                "what medications do you take" in med_q.lower()
            ]
            is_generic = any(generic_indicators)
            
            assert_test(
                is_symptom_specific and not is_generic,
                f"Medication question is symptom-specific (Q{q_num}): '{med_q}'",
                test_name
            )
            results["medication_wording"] = {
                "passed": is_symptom_specific and not is_generic,
                "question": med_q,
                "is_symptom_specific": is_symptom_specific,
                "is_generic": is_generic
            }
        else:
            warn_test("No medication questions found", test_name)
            results["medication_wording"] = {"passed": False, "reason": "No medication question found"}
    
    # ========================================================================
    # CHECK 3: Travel Question Prevention
    # ========================================================================
    if checks.get("check_no_travel") or not recently_travelled:
        travel_keywords = ["travel", "viaj", "traveled", "viajado", "trip", "viaje", "journey"]
        travel_questions = [
            (i+1, q) for i, q in enumerate(questions)
            if any(keyword in q.lower() for keyword in travel_keywords)
        ]
        
        assert_test(
            len(travel_questions) == 0,
            f"No travel questions when recently_travelled=False. Found: {[f'Q{i}: {q[:50]}...' for i, q in travel_questions]}",
            test_name
        )
        results["travel_prevention"] = {
            "passed": len(travel_questions) == 0,
            "travel_questions": travel_questions
        }
    
    # ========================================================================
    # CHECK 4: Chronic Monitoring Questions
    # ========================================================================
    if checks.get("check_chronic_monitoring"):
        monitoring_keywords = checks.get(
            "monitoring_keywords", 
            ["monitor", "reading", "check", "screening", "test", "exam", 
             "peak flow", "inhaler", "glucose", "blood sugar", "blood pressure", 
             "bp", "hba1c", "screening", "examen", "monitoreo"]
        )
        
        monitoring_questions = [
            (i+1, q) for i, q in enumerate(questions)
            if any(keyword in q.lower() for keyword in monitoring_keywords)
        ]
        
        # Check if appears early (within first 5 questions)
        early_monitoring = [
            (i+1, q) for i, q in enumerate(questions[:5])
            if any(keyword in q.lower() for keyword in monitoring_keywords)
        ]
        
        assert_test(
            len(monitoring_questions) > 0,
            f"Chronic monitoring questions found: {[f'Q{i}: {q[:60]}...' for i, q in monitoring_questions]}",
            test_name
        )
        
        if len(monitoring_questions) > 0:
            assert_test(
                len(early_monitoring) > 0,
                f"Monitoring question appears early (within first 5). Found at: {[f'Q{i}' for i, _ in early_monitoring]}",
                test_name
            )
        
        results["chronic_monitoring"] = {
            "passed": len(monitoring_questions) > 0 and len(early_monitoring) > 0,
            "monitoring_questions": monitoring_questions,
            "early_monitoring": early_monitoring
        }
    
    # ========================================================================
    # CHECK 5: Question Ordering
    # ========================================================================
    if checks.get("check_ordering"):
        # Find indices of key question types
        duration_idx = next((
            i for i, q in enumerate(questions) 
            if any(word in q.lower() for word in [
                "how long", "duration", "cuánto tiempo", "duración", 
                "when did", "cuándo", "since when", "desde cuándo"
            ])
        ), -1)
        
        medication_idx = next((
            i for i, q in enumerate(questions) 
            if any(word in q.lower() for word in [
                "medication", "medicine", "medicamento", 
                "treatment", "tratamiento", "remedy", "remedio"
            ])
        ), -1)
        
        if duration_idx >= 0 and medication_idx >= 0:
            assert_test(
                duration_idx < medication_idx,
                f"Duration (Q{duration_idx+1}) comes before medication (Q{medication_idx+1})",
                test_name
            )
            results["ordering"] = {
                "passed": duration_idx < medication_idx,
                "duration_idx": duration_idx,
                "medication_idx": medication_idx
            }
        else:
            warn_test("Could not verify ordering (missing duration or medication question)", test_name)
            results["ordering"] = {"passed": None, "reason": "Missing questions"}
    
    # ========================================================================
    # CHECK 6: Safety Constraints
    # ========================================================================
    if checks.get("check_safety_constraints"):
        # Check for inappropriate questions based on gender/age
        if patient_gender.lower() in ["male", "m", "masculino", "hombre"]:
            menstrual_questions = [
                (i+1, q) for i, q in enumerate(questions)
                if any(keyword in q.lower() for keyword in [
                    "menstrual", "period", "menstruation", "menstruo", 
                    "periodo", "menstruación", "regla", "ciclo menstrual"
                ])
            ]
            assert_test(
                len(menstrual_questions) == 0,
                f"No menstrual questions for male patient. Found: {[f'Q{i}' for i, _ in menstrual_questions]}",
                test_name
            )
            results["safety_male"] = {
                "passed": len(menstrual_questions) == 0,
                "menstrual_questions": menstrual_questions
            }
        
        if patient_age is not None and (patient_age < 12 or patient_age > 60):
            menstrual_questions = [
                (i+1, q) for i, q in enumerate(questions)
                if any(keyword in q.lower() for keyword in [
                    "menstrual", "period", "menstruation", "menstruo",
                    "periodo", "menstruación", "regla", "ciclo menstrual"
                ])
            ]
            assert_test(
                len(menstrual_questions) == 0,
                f"No menstrual questions for age {patient_age}. Found: {[f'Q{i}' for i, _ in menstrual_questions]}",
                test_name
            )
            results["safety_age"] = {
                "passed": len(menstrual_questions) == 0,
                "menstrual_questions": menstrual_questions
            }
    
    # ========================================================================
    # CHECK 7: No Duplicate Questions
    # ========================================================================
    if checks.get("check_no_duplicates"):
        duplicates = []
        for i, q1 in enumerate(questions):
            for j, q2 in enumerate(questions[i+1:], start=i+1):
                # Check for exact duplicates
                if q1.lower().strip() == q2.lower().strip():
                    duplicates.append((i+1, j+1, q1))
                # Check for very similar questions (high word overlap)
                elif len(set(q1.lower().split()) & set(q2.lower().split())) / max(len(q1.split()), len(q2.split())) > 0.8:
                    duplicates.append((i+1, j+1, f"Similar: '{q1[:50]}...' and '{q2[:50]}...'"))
        
        assert_test(
            len(duplicates) == 0,
            f"No duplicate questions. Found: {duplicates}",
            test_name
        )
        results["no_duplicates"] = {
            "passed": len(duplicates) == 0,
            "duplicates": duplicates
        }
    
    return results

async def main():
    print_header("ALL-ROUNDER COMPREHENSIVE TEST SUITE FOR INTAKE FORM")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    print("✅ Environment configured")
    
    test_scenarios = []
    
    # ============================================================================
    # CATEGORY 1: MULTI-SYMPTOM HANDLING TESTS
    # ============================================================================
    print_section("CATEGORY 1: MULTI-SYMPTOM HANDLING")
    
    test_scenarios.extend([
        {
            "name": "Multi-Symptom: Cough and Fever",
            "chief_complaint": "Cough and fever for 3 days",
            "patient_age": 35,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_multi_symptom": "cough, fever",
                "check_medication_wording": True,
                "check_no_travel": True,
                "check_ordering": True,
                "check_no_duplicates": True
            }
        },
        {
            "name": "Multi-Symptom: Headache and Nausea",
            "chief_complaint": "Headache and nausea",
            "patient_age": 28,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_multi_symptom": "headache, nausea",
                "check_medication_wording": True,
                "check_ordering": True
            }
        },
        {
            "name": "Multi-Symptom: Chest Pain and Shortness of Breath",
            "chief_complaint": "Chest pain and shortness of breath",
            "patient_age": 45,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_multi_symptom": "chest pain, shortness of breath",
                "check_medication_wording": True,
                "check_ordering": True
            }
        },
        {
            "name": "Multi-Symptom: Complex (3+ symptoms)",
            "chief_complaint": "Headache, nausea, and fatigue",
            "patient_age": 32,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_multi_symptom": "headache, nausea, fatigue",
                "check_medication_wording": True
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 2: MEDICATION QUESTION WORDING TESTS
    # ============================================================================
    print_section("CATEGORY 2: MEDICATION QUESTION WORDING")
    
    test_scenarios.extend([
        {
            "name": "Medication Wording: Headache",
            "chief_complaint": "Headache",
            "patient_age": 28,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_medication_wording": True,
                "check_ordering": True
            }
        },
        {
            "name": "Medication Wording: Back Pain",
            "chief_complaint": "Lower back pain",
            "patient_age": 42,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_medication_wording": True
            }
        },
        {
            "name": "Medication Wording: Spanish",
            "chief_complaint": "Dolor de cabeza",
            "patient_age": 30,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "sp",
            "checks": {
                "check_medication_wording": True
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 3: TRAVEL PREVENTION TESTS
    # ============================================================================
    print_section("CATEGORY 3: TRAVEL PREVENTION (Checkbox NOT Ticked)")
    
    test_scenarios.extend([
        {
            "name": "Travel Prevention: Diarrhea (No Travel)",
            "chief_complaint": "Diarrhea and stomach cramps",
            "patient_age": 29,
            "patient_gender": "male",
            "recently_travelled": False,  # CRITICAL: NOT ticked
            "language": "en",
            "checks": {
                "check_no_travel": True,
                "check_medication_wording": True
            }
        },
        {
            "name": "Travel Prevention: Fever (No Travel)",
            "chief_complaint": "Fever and chills",
            "patient_age": 35,
            "patient_gender": "male",
            "recently_travelled": False,  # NOT ticked
            "language": "en",
            "checks": {
                "check_no_travel": True
            }
        },
        {
            "name": "Travel Allowed: Diarrhea (Travel Ticked)",
            "chief_complaint": "Diarrhea and stomach cramps",
            "patient_age": 29,
            "patient_gender": "male",
            "recently_travelled": True,  # TICKED
            "language": "en",
            "checks": {
                "check_medication_wording": True
                # Note: We don't check for travel questions here as it depends on condition
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 4: CHRONIC DISEASE MONITORING TESTS
    # ============================================================================
    print_section("CATEGORY 4: CHRONIC DISEASE MONITORING")
    
    test_scenarios.extend([
        {
            "name": "Chronic Monitoring: Asthma",
            "chief_complaint": "Shortness of breath and wheezing",
            "patient_age": 28,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "max_questions": 8,
            "checks": {
                "check_chronic_monitoring": True,
                "monitoring_keywords": ["monitor", "peak flow", "inhaler", "frequency", "usage", "exacerbation"],
                "check_medication_wording": True,
                "check_ordering": True
            }
        },
        {
            "name": "Chronic Monitoring: Diabetes",
            "chief_complaint": "High blood sugar readings",
            "patient_age": 52,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "max_questions": 8,
            "checks": {
                "check_chronic_monitoring": True,
                "monitoring_keywords": ["monitor", "reading", "check", "glucose", "blood sugar", "HbA1c", "screening"],
                "check_medication_wording": True
            }
        },
        {
            "name": "Chronic Monitoring: Hypertension",
            "chief_complaint": "High blood pressure readings",
            "patient_age": 48,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "max_questions": 8,
            "checks": {
                "check_chronic_monitoring": True,
                "monitoring_keywords": ["monitor", "reading", "check", "blood pressure", "BP", "screening"],
                "check_medication_wording": True
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 5: QUESTION ORDERING TESTS
    # ============================================================================
    print_section("CATEGORY 5: QUESTION ORDERING")
    
    test_scenarios.extend([
        {
            "name": "Question Ordering: Acute Condition",
            "chief_complaint": "Chest pain",
            "patient_age": 45,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_ordering": True,
                "check_medication_wording": True,
                "check_no_duplicates": True
            }
        },
        {
            "name": "Question Ordering: Chronic Condition",
            "chief_complaint": "Asthma symptoms worsening",
            "patient_age": 35,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "max_questions": 8,
            "checks": {
                "check_ordering": True,
                "check_chronic_monitoring": True,
                "monitoring_keywords": ["monitor", "peak flow", "inhaler"]
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 6: SAFETY CONSTRAINTS TESTS
    # ============================================================================
    print_section("CATEGORY 6: SAFETY CONSTRAINTS")
    
    test_scenarios.extend([
        {
            "name": "Safety: Male Patient (No Menstrual Questions)",
            "chief_complaint": "Abdominal pain",
            "patient_age": 30,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_safety_constraints": True,
                "check_medication_wording": True
            }
        },
        {
            "name": "Safety: Age < 12 (No Menstrual Questions)",
            "chief_complaint": "Stomach pain",
            "patient_age": 8,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_safety_constraints": True
            }
        },
        {
            "name": "Safety: Age > 60 (No Menstrual Questions)",
            "chief_complaint": "Irregular bleeding",
            "patient_age": 65,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_safety_constraints": True
            }
        }
    ])
    
    # ============================================================================
    # CATEGORY 7: EDGE CASES AND COMBINATIONS
    # ============================================================================
    print_section("CATEGORY 7: EDGE CASES AND COMBINATIONS")
    
    test_scenarios.extend([
        {
            "name": "Edge Case: Multi-Symptom + Chronic Disease",
            "chief_complaint": "Cough and shortness of breath",
            "patient_age": 35,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "en",
            "max_questions": 8,
            "checks": {
                "check_multi_symptom": "cough, shortness of breath",
                "check_medication_wording": True,
                "check_no_travel": True
            }
        },
        {
            "name": "Edge Case: Spanish Language",
            "chief_complaint": "Tos y fiebre",
            "patient_age": 35,
            "patient_gender": "male",
            "recently_travelled": False,
            "language": "sp",
            "checks": {
                "check_multi_symptom": "tos, fiebre",
                "check_medication_wording": True,
                "check_no_travel": True
            }
        },
        {
            "name": "Edge Case: Single Word Symptom",
            "chief_complaint": "Fatigue",
            "patient_age": 40,
            "patient_gender": "female",
            "recently_travelled": False,
            "language": "en",
            "checks": {
                "check_medication_wording": True,
                "check_ordering": True
            }
        }
    ])
    
    # ============================================================================
    # RUN ALL TESTS
    # ============================================================================
    print_header("RUNNING ALL TEST SCENARIOS", "=")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print_test(scenario["name"], i, len(test_scenarios))
        
        result = await run_test_scenario(
            test_name=scenario["name"],
            chief_complaint=scenario["chief_complaint"],
            patient_age=scenario.get("patient_age", 35),
            patient_gender=scenario.get("patient_gender", "male"),
            recently_travelled=scenario.get("recently_travelled", False),
            language=scenario.get("language", "en"),
            max_questions=scenario.get("max_questions", 10),
            expected_checks=scenario.get("checks"),
            verbose=True
        )
        
        # Small delay between tests to avoid rate limiting
        await asyncio.sleep(1)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print_header("FINAL TEST RESULTS SUMMARY", "=")
    
    total_checks = test_results["total_tests"]
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings_count = len(test_results["warnings"])
    
    print(f"\n📊 TEST STATISTICS:")
    print(f"   Total Checks: {total_checks}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Warnings: {warnings_count}")
    
    success_rate = (passed / total_checks * 100) if total_checks > 0 else 0
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
    
    if failed > 0:
        print(f"\n❌ FAILED CHECKS ({failed}):")
        for i, fail in enumerate(test_results["failed"][:10], 1):  # Show first 10
            print(f"   {i}. {fail}")
        if len(test_results["failed"]) > 10:
            remaining = len(test_results["failed"]) - 10
            print(f"   ... and {remaining} more")
    
    if warnings_count > 0:
        print(f"\n⚠️  WARNINGS ({warnings_count}):")
        for i, warn in enumerate(test_results["warnings"][:5], 1):  # Show first 5
            print(f"   {i}. {warn}")
        if len(test_results["warnings"]) > 5:
            remaining = len(test_results["warnings"]) - 5
            print(f"   ... and {remaining} more")
    
    # Detailed breakdown by test
    print(f"\n📋 DETAILED BREAKDOWN BY TEST:")
    for test_name, details in test_results["test_details"].items():
        passed_count = len(details.get("passed", []))
        failed_count = len(details.get("failed", []))
        warn_count = len(details.get("warnings", []))
        total = passed_count + failed_count
        
        if total > 0:
            rate = (passed_count / total * 100) if total > 0 else 0
            status = "✅" if failed_count == 0 else "❌"
            print(f"   {status} {test_name}: {passed_count}/{total} passed ({rate:.0f}%)")
    
    # Final verdict
    print(f"\n{'=' * 100}")
    if success_rate >= 95:
        print("✅ EXCELLENT - System is working flawlessly!")
    elif success_rate >= 85:
        print("✅ VERY GOOD - System is working well with minor issues")
    elif success_rate >= 75:
        print("⚠️  GOOD - System is working but needs some improvements")
    else:
        print("❌ NEEDS IMPROVEMENT - Several issues detected, review required")
    print(f"{'=' * 100}\n")

if __name__ == "__main__":
    asyncio.run(main())

