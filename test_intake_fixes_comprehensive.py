"""
Comprehensive Test Suite for Intake Form Fixes

Tests all 4 critical issues:
1. Multi-symptom handling
2. Medication question wording (symptom-specific)
3. Travel questions prevention (when checkbox not ticked)
4. Chronic monitoring/screening questions for chronic diseases
5. Question ordering

Run with: python3 test_intake_fixes_comprehensive.py
"""

import asyncio
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clinicai.adapters.external.question_service_openai import OpenAIQuestionService

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def print_header(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_test(test_name: str):
    print(f"\n🧪 TEST: {test_name}")
    print("-" * 80)

def assert_test(condition: bool, message: str, test_name: str):
    if condition:
        print(f"✅ PASS: {message}")
        test_results["passed"].append(f"{test_name}: {message}")
    else:
        print(f"❌ FAIL: {message}")
        test_results["failed"].append(f"{test_name}: {message}")

def warn_test(message: str, test_name: str):
    print(f"⚠️  WARN: {message}")
    test_results["warnings"].append(f"{test_name}: {message}")

async def run_test_scenario(
    test_name: str,
    chief_complaint: str,
    patient_age: int = 35,
    patient_gender: str = "male",
    recently_travelled: bool = False,
    language: str = "en",
    max_questions: int = 10,
    expected_checks: Dict[str, Any] = None
) -> List[str]:
    """Run a test scenario and return all questions generated"""
    
    print_test(test_name)
    print(f"Chief Complaint: {chief_complaint}")
    print(f"Patient: {patient_age}y/o {patient_gender}, Recently travelled: {recently_travelled}")
    
    service = OpenAIQuestionService()
    
    asked_questions = []
    previous_answers = []
    
    # Generate first question
    first_q = await service.generate_first_question(chief_complaint, language)
    asked_questions.append(first_q)
    print(f"\nQ1: {first_q}")
    
    # Simulate first answer
    first_answer = chief_complaint
    previous_answers.append(first_answer)
    print(f"A1: {first_answer}")
    
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
            
            if next_q and next_q.strip():
                asked_questions.append(next_q)
                print(f"Q{len(asked_questions)}: {next_q}")
                
                # Simulate answer
                answer = f"Answer to Q{len(asked_questions)}"
                previous_answers.append(answer)
                print(f"A{len(asked_questions)}: {answer}")
            else:
                break
        except Exception as e:
            print(f"Error generating question: {e}")
            break
    
    print(f"\n📊 Generated {len(asked_questions)} questions total")
    
    # Run checks
    if expected_checks:
        run_checks(test_name, asked_questions, expected_checks, recently_travelled, chief_complaint)
    
    return asked_questions

def run_checks(test_name: str, questions: List[str], checks: Dict[str, Any], recently_travelled: bool, chief_complaint: str):
    """Run validation checks on generated questions"""
    
    questions_lower = [q.lower() for q in questions]
    questions_text = " ".join(questions_lower)
    
    # Check 1: Multi-symptom handling
    if checks.get("check_multi_symptom"):
        symptoms = checks["check_multi_symptom"]
        if isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(",")]
        
        all_mentioned = all(any(symptom.lower() in questions_text for symptom in symptoms) for symptom in symptoms)
        assert_test(
            all_mentioned,
            f"All symptoms mentioned in questions: {symptoms}",
            test_name
        )
    
    # Check 2: Medication question wording (symptom-specific)
    if checks.get("check_medication_wording"):
        medication_questions = [q for q in questions if any(word in q.lower() for word in ["medication", "medicine", "medicamento", "treatment", "tratamiento", "remedy", "remedio"])]
        
        if medication_questions:
            # Check if medication question is symptom-specific (mentions the chief complaint or symptom)
            symptom_specific = any(
                any(symptom.lower() in q.lower() for symptom in chief_complaint.lower().split()) or
                "for your" in q.lower() or "para su" in q.lower() or
                "for this" in q.lower() or "para esto" in q.lower() or
                "home remedy" in q.lower() or "remedio casero" in q.lower()
                for q in medication_questions
            )
            
            generic_medication = any(
                "what medications are you currently taking" in q.lower() or
                "qué medicamentos está tomando actualmente" in q.lower()
                for q in medication_questions
            )
            
            assert_test(
                symptom_specific and not generic_medication,
                f"Medication question is symptom-specific (not generic). Found: {medication_questions[0]}",
                test_name
            )
        else:
            warn_test("No medication questions found", test_name)
    
    # Check 3: Travel questions prevention
    if checks.get("check_no_travel") or not recently_travelled:
        travel_questions = [q for q in questions if any(word in q.lower() for word in ["travel", "viaj", "traveled", "viajado", "trip", "viaje"])]
        assert_test(
            len(travel_questions) == 0,
            f"No travel questions when recently_travelled=False. Found: {travel_questions}",
            test_name
        )
    
    # Check 4: Chronic monitoring questions
    if checks.get("check_chronic_monitoring"):
        monitoring_keywords = checks.get("monitoring_keywords", ["monitor", "reading", "check", "screening", "test", "exam", "peak flow", "inhaler", "glucose", "blood sugar", "blood pressure", "bp", "hbA1c"])
        monitoring_questions = [
            q for q in questions 
            if any(keyword in q.lower() for keyword in monitoring_keywords)
        ]
        
        # Check if monitoring question appears early (within first 5 questions)
        early_monitoring = [
            (i+1, q) for i, q in enumerate(questions[:5])
            if any(keyword in q.lower() for keyword in monitoring_keywords)
        ]
        
        assert_test(
            len(monitoring_questions) > 0,
            f"Chronic monitoring questions found. Keywords: {monitoring_keywords}. Found: {monitoring_questions}",
            test_name
        )
        
        if len(monitoring_questions) > 0:
            assert_test(
                len(early_monitoring) > 0,
                f"Chronic monitoring question appears early (within first 5 questions). Found at: {[f'Q{i}' for i, _ in early_monitoring]}",
                test_name
            )
    
    # Check 5: Question ordering
    if checks.get("check_ordering"):
        # Check if duration comes before medications
        duration_idx = next((i for i, q in enumerate(questions) if any(word in q.lower() for word in ["how long", "duration", "cuánto tiempo", "duración", "when did", "cuándo"])), -1)
        medication_idx = next((i for i, q in enumerate(questions) if any(word in q.lower() for word in ["medication", "medicine", "medicamento", "treatment", "tratamiento"])), -1)
        
        if duration_idx >= 0 and medication_idx >= 0:
            assert_test(
                duration_idx < medication_idx,
                f"Duration question (Q{duration_idx+1}) comes before medication question (Q{medication_idx+1})",
                test_name
            )

async def main():
    print_header("COMPREHENSIVE INTAKE FORM FIXES TEST SUITE")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    print("✅ Environment configured")
    
    # ============================================================================
    # TEST 1: Multi-Symptom Handling
    # ============================================================================
    await run_test_scenario(
        test_name="Test 1: Multi-Symptom Handling",
        chief_complaint="Cough and fever for 3 days",
        patient_age=35,
        patient_gender="male",
        recently_travelled=False,
        expected_checks={
            "check_multi_symptom": "cough, fever",
            "check_medication_wording": True,
            "check_no_travel": True
        }
    )
    
    # ============================================================================
    # TEST 2: Medication Question Wording
    # ============================================================================
    await run_test_scenario(
        test_name="Test 2: Medication Question Wording",
        chief_complaint="Headache",
        patient_age=28,
        patient_gender="female",
        recently_travelled=False,
        expected_checks={
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 3: Travel Prevention (Checkbox Not Ticked)
    # ============================================================================
    await run_test_scenario(
        test_name="Test 3: Travel Prevention (Checkbox Not Ticked)",
        chief_complaint="Diarrhea and stomach cramps",
        patient_age=29,
        patient_gender="male",
        recently_travelled=False,  # CRITICAL: Checkbox NOT ticked
        expected_checks={
            "check_no_travel": True,
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 4: Chronic Disease Monitoring
    # ============================================================================
    await run_test_scenario(
        test_name="Test 4: Chronic Disease - Asthma Monitoring",
        chief_complaint="Shortness of breath and wheezing",
        patient_age=28,
        patient_gender="female",
        recently_travelled=False,
        expected_checks={
            "check_chronic_monitoring": True,
            "monitoring_keywords": ["monitor", "peak flow", "inhaler", "frequency", "usage", "exacerbation"],
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 5: Chronic Disease - Diabetes Monitoring
    # ============================================================================
    await run_test_scenario(
        test_name="Test 5: Chronic Disease - Diabetes Monitoring",
        chief_complaint="High blood sugar readings",
        patient_age=52,
        patient_gender="female",
        recently_travelled=False,
        expected_checks={
            "check_chronic_monitoring": True,
            "monitoring_keywords": ["monitor", "reading", "check", "glucose", "blood sugar", "HbA1c", "screening"],
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 6: Chronic Disease - Hypertension Monitoring
    # ============================================================================
    await run_test_scenario(
        test_name="Test 6: Chronic Disease - Hypertension Monitoring",
        chief_complaint="High blood pressure readings",
        patient_age=48,
        patient_gender="male",
        recently_travelled=False,
        expected_checks={
            "check_chronic_monitoring": True,
            "monitoring_keywords": ["monitor", "reading", "check", "blood pressure", "BP", "screening"],
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 7: Question Ordering
    # ============================================================================
    await run_test_scenario(
        test_name="Test 7: Question Ordering",
        chief_complaint="Chest pain",
        patient_age=45,
        patient_gender="male",
        recently_travelled=False,
        expected_checks={
            "check_ordering": True,
            "check_medication_wording": True
        }
    )
    
    # ============================================================================
    # TEST 8: Multi-Symptom + Chronic Disease
    # ============================================================================
    await run_test_scenario(
        test_name="Test 8: Multi-Symptom + Chronic Disease",
        chief_complaint="Cough and shortness of breath",
        patient_age=35,
        patient_gender="male",
        recently_travelled=False,
        expected_checks={
            "check_multi_symptom": "cough, shortness of breath",
            "check_medication_wording": True,
            "check_no_travel": True
        }
    )
    
    # ============================================================================
    # TEST 9: Travel Allowed (Checkbox Ticked)
    # ============================================================================
    await run_test_scenario(
        test_name="Test 9: Travel Allowed (Checkbox Ticked)",
        chief_complaint="Diarrhea and stomach cramps",
        patient_age=29,
        patient_gender="male",
        recently_travelled=True,  # Checkbox TICKED
        expected_checks={
            "check_medication_wording": True
            # Note: We don't check for travel questions here since it depends on condition
        }
    )
    
    # ============================================================================
    # TEST 10: Complex Multi-Symptom Case
    # ============================================================================
    await run_test_scenario(
        test_name="Test 10: Complex Multi-Symptom Case",
        chief_complaint="Headache, nausea, and fatigue",
        patient_age=32,
        patient_gender="female",
        recently_travelled=False,
        expected_checks={
            "check_multi_symptom": "headache, nausea, fatigue",
            "check_medication_wording": True,
            "check_no_travel": True
        }
    )
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print_header("TEST RESULTS SUMMARY")
    
    total_tests = len(test_results["passed"]) + len(test_results["failed"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings_count = len(test_results["warnings"])
    
    print(f"\n📊 Total Checks: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings_count}")
    
    if failed > 0:
        print("\n❌ FAILED CHECKS:")
        for fail in test_results["failed"]:
            print(f"  - {fail}")
    
    if warnings_count > 0:
        print("\n⚠️  WARNINGS:")
        for warn in test_results["warnings"]:
            print(f"  - {warn}")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n✅ EXCELLENT - System is working as expected!")
    elif success_rate >= 75:
        print("\n⚠️  GOOD - Minor issues detected")
    else:
        print("\n❌ NEEDS IMPROVEMENT - Several issues detected")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

