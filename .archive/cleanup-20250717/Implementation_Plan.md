# MemMimic - Implementation Plan

**Version:** 1.0
**Date:** 2025-07-17
**Manager Agent:** Gemini 2.5 Pro (Cursor)
**Project Principal:** User

## 1. Project Overview and Core Objectives

This plan outlines the strategic implementation for enhancing the MemMimic Active Memory System. The primary objectives are:
1.  **Remediate Critical QA Deficiencies:** Stabilize the project by refactoring the entire testing suite to modern standards, ensuring reliability and enabling robust CI/CD integration.
2.  **Complete the Active Memory Management System (AMMS):** Finalize all functional requirements for the AMMS as specified in the `docs/PRD_ActiveMemorySystem.md`.
3.  **Harden System Integrations:** Resolve all identified inconsistencies in API definitions, external service configurations, and hardcoded paths to improve system integrity and portability.
4.  **Operationalize Advanced AI Concepts:** Translate abstract philosophical directives and AIDE workflows into measurable, practical outcomes.

## 2. Memory Bank Configuration

Based on the project's complexity and parallel workstreams, we have adopted a **Directory-Based Memory Bank System**, located in the root `/Memory` directory.

- **Structure:** Each primary phase of this implementation plan will have its own subdirectory within `/Memory` (e.g., `/Memory/Phase_01_QA_Refactoring/`).
- **Log Naming Convention:** Individual task logs will be named `Task_XX_Description_Log.md` within the appropriate phase directory.
- **Rationale:** This structure provides clear separation of concerns, simplifies review, and scales effectively with the project's phased approach. All logging activities must adhere to the format defined in `agentic-project-management/prompts/02_Utility_Prompts_And_Format_Definitions/Memory_Bank_Log_Format.md`.

## 3. Implementation Phases

---

### **Phase 1: QA Foundation Refactoring (Top Priority)**

**Objective:** To establish a reliable, automated, and standardized testing foundation for the entire project.

*   **Task 1.1: Migrate `test_active_memory.py` to `pytest`**
    *   **Sub-Task 1.1.1:** Refactor all test functions to use standard `pytest` assertions (`assert`) instead of `print` statements and `try/except` blocks.
    *   **Sub-Task 1.1.2:** Create a `pytest` fixture to provide a temporary, isolated file-based or in-memory SQLite database for test functions.
    *   **Sub-Task 1.1.3:** Replace all hardcoded database paths (`memmimic_memories.db`) in the test with the new database fixture.
    *   **Sub-Task 1.1.4:** Ensure the refactored test can be discovered and run successfully by the `pytest` runner.

*   **Task 1.2: Migrate `test_comprehensive.py` to `pytest`**
    *   **Sub-Task 1.2.1:** Break down the monolithic script into smaller, more focused `pytest` test files (e.g., `test_system_env.py`, `test_api_tools.py`, `test_mcp_integration.py`, `test_performance.py`).
    *   **Sub-Task 1.2.2:** Convert all test logic to use `pytest` assertions and fixtures for setup/teardown (e.g., for creating temporary databases and starting/stopping the MCP server).
    *   **Sub-Task 1.2.3:** Replace custom reporting logic with standard `pytest` output.

*   **Task 1.3: Migrate `test_cxd_integration.py` and `test_unified_api.py`**
    *   **Sub-Task 1.3.1:** Combine and refactor these scripts into a single, cohesive `pytest` integration test file.
    *   **Sub-Task 1.3.2:** Replace all `print` statements with `pytest` assertions.
    *   **Sub-Task 1.3.3:** Develop deeper functional tests for each API tool, going beyond simple existence checks.

*   **Task 1.4: Integrate Code Coverage Reporting**
    *   **Sub-Task 1.4.1:** Add `pytest-cov` as a development dependency.
    *   **Sub-Task 1.4.2:** Configure `pytest` to generate a code coverage report (HTML and terminal).
    *   **Sub-Task 1.4.3:** Set an initial coverage target (e.g., 80%) and create a plan to address any major gaps.

---

### **Phase 2: Active Memory Management System (AMMS) Completion**

**Objective:** To finalize and validate all AMMS features as per the PRD.

*   **Task 2.1: Implement Memory Archival and Pruning System**
    *   **Sub-Task 2.1.1:** Develop the archival mechanism for moving low-ranking memories from the active database to a separate archive store/table (`FR-010`).
    *   **Sub-Task 2.1.2:** Implement the pruning logic for permanently deleting memories that fall below the prune threshold.
    *   **Sub-Task 2.1.3:** Write comprehensive unit and integration tests for archival and pruning.

*   **Task 2.2: Implement Archived Memory Recovery**
    *   **Sub-Task 2.2.1:** Develop the mechanism to search the archive and restore relevant memories back into the active pool when needed (`FR-011`).
    *   **Sub-Task 2.2.2:** Test the recovery mechanism under various scenarios.

*   **Task 2.3: Implement Manual Override and Policy Management**
    *   **Sub-Task 2.3.1:** Implement the manual override capability to protect specific memories from automated cleanup (`FR-012`).
    *   **Sub-Task 2.3.2:** Ensure retention policies from the PRD (e.g., for `synthetic_wisdom`) are correctly implemented.

*   **Task 2.4: Performance Validation**
    *   **Sub-Task 2.4.1:** Using the new `pytest` performance test suite, validate the AMMS against the success metrics in the PRD (e.g., query time, memory usage).

---

### **Phase 3: API, Integration, and Configuration Hardening**

**Objective:** To eliminate inconsistencies and improve the overall robustness and portability of the system.

*   **Task 3.1: Resolve `greptile` Tool Discrepancy**
    *   **Sub-Task 3.1.1:** Investigate the purpose of the `greptile` tools listed in `.claude/settings.local.json`.
    *   **Sub-Task 3.1.2:** If they are legacy, remove them from the configuration. If they are active but separate, document their integration. Ensure the configuration accurately reflects the state of the system.

*   **Task 3.2: Formalize API Consistency**
    *   **Sub-Task 3.2.1:** Conduct a line-by-line review of the MCP API in `server.js` and the Python API in `api.py`.
    *   **Sub-Task 3.2.2:** Implement any stubbed-out functionalities in `api.py` (e.g., `socratic_dialogue`, `context_tale`) or mark them clearly as "Not Implemented" in the external API schema description.

*   **Task 3.3: Externalize Hardcoded Paths**
    *   **Sub-Task 3.3.1:** Refactor `src/memmimic/mcp/server.js` to use an environment variable for the Python `venv` path instead of a hardcoded relative path.
    *   **Sub-Task 3.3.2:** Refactor `git_helper.py` to use relative paths or dynamically determine the project root instead of a hardcoded CWD.

---

### **Phase 4: Operationalizing Philosophical & AIDE Components**

**Objective:** To bridge the gap between abstract concepts and practical, measurable functionality.

*   **Task 4.1: Develop Metrics for Philosophical "Tales"**
    *   **Sub-Task 4.1.1:** Research and define a set of qualitative or quantitative metrics to assess the impact of core "tales" on Claude's responses (e.g., improved contextual awareness, adherence to defined principles).
    *   **Sub-Task 4.1.2:** Implement a mechanism to track these metrics over time.

*   **Task 4.2: Enhance AIDE Workflows**
    *   **Sub-Task 4.2.1:** Review the existing AIDE scripts (`git_helper.py`, `commit_summary.md` process) for potential improvements in robustness and user experience.
    *   **Sub-Task 4.2.2:** Explore further integration of the `.taskmaster` system into the development lifecycle.

## 4. Dependencies and Task Sequencing

- **Phase 1 is a blocker for all subsequent phases.** A stable testing environment is required before adding or modifying features.
- Within Phase 1, `Task 1.1` and `1.2` can be done in parallel. `Task 1.4` depends on the completion of the other tasks in this phase.
- **Phase 2** tasks depend on the completion of Phase 1.
- **Phase 3 and 4** can be worked on in parallel with Phase 2, but any code changes must adhere to the new testing standards established in Phase 1. 