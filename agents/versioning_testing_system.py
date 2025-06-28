#!/usr/bin/env python3
"""
Versioning & Testing System for AgentAlina
Implements GitOps-based versioning, automated testing, A/B testing,
and continuous integration for prompt templates and agent configurations.
"""

import asyncio
import logging
import json
import os
import subprocess
import tempfile
import shutil
import hashlib
import yaml
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import git
import pytest
import coverage
from unittest.mock import Mock, patch
import statistics
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class DeploymentStage(Enum):
    """Deployment stages."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    AB_TEST = "ab_test"
    REGRESSION = "regression"
    SMOKE = "smoke"

@dataclass
class Version:
    """Represents a version of a component."""
    id: str
    component: str
    version: str
    commit_hash: str
    created_at: datetime
    created_by: str
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class TestResult:
    """Represents a test execution result."""
    id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    version_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    output: str
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['test_type'] = self.test_type.value
        data['status'] = self.status.value
        data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    name: str
    description: str
    control_version: str
    treatment_versions: List[str]
    traffic_split: Dict[str, float]  # version -> percentage
    success_metrics: List[str]
    duration_hours: int
    min_sample_size: int
    significance_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class GitOpsManager:
    """Manages GitOps operations for versioning."""
    
    def __init__(self, repo_path: str, remote_url: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.remote_url = remote_url
        self.repo = None
        
        # Initialize or open repository
        self._init_repository()
    
    def _init_repository(self):
        """Initialize or open Git repository."""
        try:
            if self.repo_path.exists() and (self.repo_path / '.git').exists():
                self.repo = git.Repo(self.repo_path)
                logger.info(f"Opened existing repository at {self.repo_path}")
            else:
                self.repo_path.mkdir(parents=True, exist_ok=True)
                self.repo = git.Repo.init(self.repo_path)
                logger.info(f"Initialized new repository at {self.repo_path}")
                
                # Add remote if provided
                if self.remote_url:
                    try:
                        self.repo.create_remote('origin', self.remote_url)
                    except git.exc.GitCommandError:
                        pass  # Remote might already exist
        
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise
    
    def create_version(self, component: str, files: Dict[str, str], 
                      description: str, author: str = "AgentAlina") -> Version:
        """Create a new version by committing files."""
        try:
            # Create component directory if it doesn't exist
            component_dir = self.repo_path / component
            component_dir.mkdir(parents=True, exist_ok=True)
            
            # Write files
            for filename, content in files.items():
                file_path = component_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding='utf-8')
            
            # Stage files
            self.repo.index.add([str(component_dir / filename) for filename in files.keys()])
            
            # Create commit
            commit = self.repo.index.commit(
                message=f"{component}: {description}",
                author=git.Actor(author, f"{author}@agentalina.ai")
            )
            
            # Generate version
            version_id = f"{component}-{commit.hexsha[:8]}-{int(datetime.now().timestamp())}"
            version = Version(
                id=version_id,
                component=component,
                version=commit.hexsha[:8],
                commit_hash=commit.hexsha,
                created_at=datetime.now(),
                created_by=author,
                description=description
            )
            
            logger.info(f"Created version {version_id} for component {component}")
            return version
        
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[Version]:
        """Get version information by ID."""
        try:
            # Parse version ID to extract commit hash
            parts = version_id.split('-')
            if len(parts) < 2:
                return None
            
            commit_hash = parts[1]
            
            # Find commit
            for commit in self.repo.iter_commits():
                if commit.hexsha.startswith(commit_hash):
                    return Version(
                        id=version_id,
                        component=parts[0],
                        version=commit.hexsha[:8],
                        commit_hash=commit.hexsha,
                        created_at=datetime.fromtimestamp(commit.committed_date),
                        created_by=commit.author.name,
                        description=commit.message.strip()
                    )
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting version {version_id}: {e}")
            return None
    
    def get_component_files(self, component: str, version_id: Optional[str] = None) -> Dict[str, str]:
        """Get files for a component at a specific version."""
        try:
            if version_id:
                version = self.get_version(version_id)
                if not version:
                    raise ValueError(f"Version {version_id} not found")
                commit = self.repo.commit(version.commit_hash)
            else:
                commit = self.repo.head.commit
            
            component_dir = Path(component)
            files = {}
            
            for item in commit.tree.traverse():
                if item.type == 'blob':  # It's a file
                    file_path = Path(item.path)
                    if file_path.parts[0] == component:
                        relative_path = str(file_path.relative_to(component_dir))
                        files[relative_path] = item.data_stream.read().decode('utf-8')
            
            return files
        
        except Exception as e:
            logger.error(f"Error getting component files: {e}")
            return {}
    
    def list_versions(self, component: Optional[str] = None, limit: int = 50) -> List[Version]:
        """List versions, optionally filtered by component."""
        try:
            versions = []
            
            for commit in self.repo.iter_commits(max_count=limit):
                message = commit.message.strip()
                if ':' in message:
                    comp, desc = message.split(':', 1)
                    comp = comp.strip()
                    desc = desc.strip()
                    
                    if component is None or comp == component:
                        version_id = f"{comp}-{commit.hexsha[:8]}-{commit.committed_date}"
                        versions.append(Version(
                            id=version_id,
                            component=comp,
                            version=commit.hexsha[:8],
                            commit_hash=commit.hexsha,
                            created_at=datetime.fromtimestamp(commit.committed_date),
                            created_by=commit.author.name,
                            description=desc
                        ))
            
            return versions
        
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
    
    def create_branch(self, branch_name: str, from_commit: Optional[str] = None) -> bool:
        """Create a new branch."""
        try:
            if from_commit:
                commit = self.repo.commit(from_commit)
                self.repo.create_head(branch_name, commit)
            else:
                self.repo.create_head(branch_name)
            
            logger.info(f"Created branch {branch_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating branch {branch_name}: {e}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """Switch to a branch."""
        try:
            self.repo.heads[branch_name].checkout()
            logger.info(f"Switched to branch {branch_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error switching to branch {branch_name}: {e}")
            return False

class TestRunner:
    """Runs various types of tests."""
    
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, TestResult] = {}
    
    async def run_unit_tests(self, component: str, version_id: str, 
                           test_files: List[str] = None) -> TestResult:
        """Run unit tests for a component."""
        test_id = f"unit-{component}-{version_id}-{int(datetime.now().timestamp())}"
        
        result = TestResult(
            id=test_id,
            test_name=f"Unit tests for {component}",
            test_type=TestType.UNIT,
            status=TestStatus.RUNNING,
            version_id=version_id,
            started_at=datetime.now(),
            completed_at=None,
            duration=None,
            output=""
        )
        
        self.results[test_id] = result
        
        try:
            # Prepare test environment
            test_env = self._prepare_test_environment(component, version_id)
            
            # Run pytest
            if test_files:
                test_paths = [str(self.test_dir / f) for f in test_files]
            else:
                test_paths = [str(self.test_dir / f"test_{component}.py")]
            
            # Run tests with coverage
            cov = coverage.Coverage()
            cov.start()
            
            pytest_args = [
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file={test_env}/test_report.json"
            ] + test_paths
            
            exit_code = pytest.main(pytest_args)
            
            cov.stop()
            cov.save()
            
            # Get coverage report
            coverage_report = self._get_coverage_report(cov)
            
            # Read test report
            report_file = Path(test_env) / "test_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    test_report = json.load(f)
            else:
                test_report = {}
            
            # Update result
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.PASSED if exit_code == 0 else TestStatus.FAILED
            result.output = json.dumps(test_report, indent=2)
            result.metrics = {
                'coverage': coverage_report,
                'exit_code': exit_code,
                'tests_run': test_report.get('summary', {}).get('total', 0),
                'tests_passed': test_report.get('summary', {}).get('passed', 0),
                'tests_failed': test_report.get('summary', {}).get('failed', 0)
            }
            
            logger.info(f"Unit tests completed for {component}: {result.status.value}")
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            logger.error(f"Error running unit tests for {component}: {e}")
        
        return result
    
    async def run_integration_tests(self, components: List[str], version_ids: List[str]) -> TestResult:
        """Run integration tests across multiple components."""
        test_id = f"integration-{'-'.join(components)}-{int(datetime.now().timestamp())}"
        
        result = TestResult(
            id=test_id,
            test_name=f"Integration tests for {', '.join(components)}",
            test_type=TestType.INTEGRATION,
            status=TestStatus.RUNNING,
            version_id="|".join(version_ids),
            started_at=datetime.now(),
            completed_at=None,
            duration=None,
            output=""
        )
        
        self.results[test_id] = result
        
        try:
            # Prepare integration test environment
            test_env = self._prepare_integration_environment(components, version_ids)
            
            # Run integration tests
            test_files = [str(self.test_dir / "integration" / f"test_{comp}_integration.py") 
                         for comp in components]
            
            pytest_args = [
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file={test_env}/integration_report.json"
            ] + test_files
            
            exit_code = pytest.main(pytest_args)
            
            # Read test report
            report_file = Path(test_env) / "integration_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    test_report = json.load(f)
            else:
                test_report = {}
            
            # Update result
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.PASSED if exit_code == 0 else TestStatus.FAILED
            result.output = json.dumps(test_report, indent=2)
            result.metrics = {
                'exit_code': exit_code,
                'tests_run': test_report.get('summary', {}).get('total', 0),
                'tests_passed': test_report.get('summary', {}).get('passed', 0),
                'tests_failed': test_report.get('summary', {}).get('failed', 0)
            }
            
            logger.info(f"Integration tests completed: {result.status.value}")
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            logger.error(f"Error running integration tests: {e}")
        
        return result
    
    async def run_performance_tests(self, component: str, version_id: str, 
                                  load_config: Dict[str, Any]) -> TestResult:
        """Run performance tests for a component."""
        test_id = f"performance-{component}-{version_id}-{int(datetime.now().timestamp())}"
        
        result = TestResult(
            id=test_id,
            test_name=f"Performance tests for {component}",
            test_type=TestType.PERFORMANCE,
            status=TestStatus.RUNNING,
            version_id=version_id,
            started_at=datetime.now(),
            completed_at=None,
            duration=None,
            output=""
        )
        
        self.results[test_id] = result
        
        try:
            # Simulate performance testing
            duration = load_config.get('duration_seconds', 60)
            concurrent_users = load_config.get('concurrent_users', 10)
            requests_per_second = load_config.get('requests_per_second', 100)
            
            # Mock performance metrics
            await asyncio.sleep(min(duration, 5))  # Simulate test execution
            
            # Generate mock performance results
            response_times = [0.1 + (i * 0.01) for i in range(100)]  # Mock data
            throughput = requests_per_second * 0.95  # Mock 95% of target
            error_rate = 0.02  # Mock 2% error rate
            
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.PASSED
            result.metrics = {
                'avg_response_time': statistics.mean(response_times),
                'p95_response_time': sorted(response_times)[94],
                'p99_response_time': sorted(response_times)[98],
                'throughput_rps': throughput,
                'error_rate': error_rate,
                'concurrent_users': concurrent_users,
                'total_requests': int(throughput * duration)
            }
            
            logger.info(f"Performance tests completed for {component}: {result.status.value}")
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            logger.error(f"Error running performance tests for {component}: {e}")
        
        return result
    
    def _prepare_test_environment(self, component: str, version_id: str) -> str:
        """Prepare test environment for a component."""
        test_env_dir = tempfile.mkdtemp(prefix=f"test_{component}_")
        
        # Create test configuration
        config = {
            'component': component,
            'version_id': version_id,
            'test_env': test_env_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = Path(test_env_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return test_env_dir
    
    def _prepare_integration_environment(self, components: List[str], version_ids: List[str]) -> str:
        """Prepare integration test environment."""
        test_env_dir = tempfile.mkdtemp(prefix="integration_test_")
        
        # Create integration test configuration
        config = {
            'components': components,
            'version_ids': version_ids,
            'test_env': test_env_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = Path(test_env_dir) / "integration_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return test_env_dir
    
    def _get_coverage_report(self, cov: coverage.Coverage) -> Dict[str, Any]:
        """Get coverage report data."""
        try:
            # Get coverage data
            total_coverage = cov.report(show_missing=False, skip_covered=False)
            
            # Get detailed coverage info
            coverage_data = cov.get_data()
            files_coverage = {}
            
            for filename in coverage_data.measured_files():
                analysis = cov.analysis2(filename)
                if analysis:
                    executed_lines = len(analysis[1])
                    missing_lines = len(analysis[3])
                    total_lines = executed_lines + missing_lines
                    
                    if total_lines > 0:
                        file_coverage = (executed_lines / total_lines) * 100
                        files_coverage[filename] = {
                            'coverage_percent': file_coverage,
                            'executed_lines': executed_lines,
                            'missing_lines': missing_lines,
                            'total_lines': total_lines
                        }
            
            return {
                'total_coverage': total_coverage,
                'files': files_coverage
            }
        
        except Exception as e:
            logger.error(f"Error getting coverage report: {e}")
            return {'error': str(e)}
    
    def get_test_result(self, test_id: str) -> Optional[TestResult]:
        """Get test result by ID."""
        return self.results.get(test_id)
    
    def get_test_results(self, component: Optional[str] = None, 
                        test_type: Optional[TestType] = None) -> List[TestResult]:
        """Get test results, optionally filtered."""
        results = list(self.results.values())
        
        if component:
            results = [r for r in results if component in r.test_name]
        
        if test_type:
            results = [r for r in results if r.test_type == test_type]
        
        return sorted(results, key=lambda r: r.started_at, reverse=True)

class ABTestManager:
    """Manages A/B testing for different versions."""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.traffic_assignments: Dict[str, str] = {}  # user_id -> version
    
    def create_ab_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test."""
        try:
            # Validate traffic split
            total_traffic = sum(config.traffic_split.values())
            if abs(total_traffic - 1.0) > 0.01:
                raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
            
            # Validate versions
            all_versions = [config.control_version] + config.treatment_versions
            for version in config.traffic_split.keys():
                if version not in all_versions:
                    raise ValueError(f"Version {version} in traffic split not in test versions")
            
            self.active_tests[config.name] = config
            self.test_results[config.name] = {
                'start_time': datetime.now().isoformat(),
                'metrics': {version: {} for version in all_versions},
                'assignments': {},
                'sample_sizes': {version: 0 for version in all_versions}
            }
            
            logger.info(f"Created A/B test: {config.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating A/B test {config.name}: {e}")
            return False
    
    def assign_user_to_version(self, test_name: str, user_id: str) -> Optional[str]:
        """Assign a user to a version for A/B testing."""
        if test_name not in self.active_tests:
            return None
        
        # Check if user already assigned
        assignment_key = f"{test_name}:{user_id}"
        if assignment_key in self.traffic_assignments:
            return self.traffic_assignments[assignment_key]
        
        config = self.active_tests[test_name]
        
        # Use hash of user_id for consistent assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        random_value = (user_hash % 10000) / 10000.0
        
        # Assign based on traffic split
        cumulative = 0.0
        for version, percentage in config.traffic_split.items():
            cumulative += percentage
            if random_value <= cumulative:
                self.traffic_assignments[assignment_key] = version
                self.test_results[test_name]['assignments'][user_id] = version
                self.test_results[test_name]['sample_sizes'][version] += 1
                return version
        
        # Fallback to control
        control_version = config.control_version
        self.traffic_assignments[assignment_key] = control_version
        self.test_results[test_name]['assignments'][user_id] = control_version
        self.test_results[test_name]['sample_sizes'][control_version] += 1
        return control_version
    
    def record_metric(self, test_name: str, user_id: str, metric_name: str, value: float):
        """Record a metric value for A/B testing."""
        if test_name not in self.active_tests:
            return
        
        assignment_key = f"{test_name}:{user_id}"
        if assignment_key not in self.traffic_assignments:
            return
        
        version = self.traffic_assignments[assignment_key]
        
        if version not in self.test_results[test_name]['metrics']:
            self.test_results[test_name]['metrics'][version] = {}
        
        if metric_name not in self.test_results[test_name]['metrics'][version]:
            self.test_results[test_name]['metrics'][version][metric_name] = []
        
        self.test_results[test_name]['metrics'][version][metric_name].append(value)
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_name not in self.active_tests:
            return {'error': f'Test {test_name} not found'}
        
        config = self.active_tests[test_name]
        results = self.test_results[test_name]
        
        analysis = {
            'test_name': test_name,
            'config': config.to_dict(),
            'start_time': results['start_time'],
            'sample_sizes': results['sample_sizes'],
            'metrics_analysis': {}
        }
        
        # Analyze each success metric
        for metric_name in config.success_metrics:
            metric_analysis = {'versions': {}}
            
            for version in [config.control_version] + config.treatment_versions:
                version_metrics = results['metrics'].get(version, {}).get(metric_name, [])
                
                if version_metrics:
                    metric_analysis['versions'][version] = {
                        'count': len(version_metrics),
                        'mean': statistics.mean(version_metrics),
                        'median': statistics.median(version_metrics),
                        'std_dev': statistics.stdev(version_metrics) if len(version_metrics) > 1 else 0
                    }
                else:
                    metric_analysis['versions'][version] = {
                        'count': 0,
                        'mean': 0,
                        'median': 0,
                        'std_dev': 0
                    }
            
            # Calculate statistical significance (simplified)
            control_metrics = results['metrics'].get(config.control_version, {}).get(metric_name, [])
            
            for treatment_version in config.treatment_versions:
                treatment_metrics = results['metrics'].get(treatment_version, {}).get(metric_name, [])
                
                if len(control_metrics) > 10 and len(treatment_metrics) > 10:
                    # Simplified t-test (would use scipy.stats in production)
                    control_mean = statistics.mean(control_metrics)
                    treatment_mean = statistics.mean(treatment_metrics)
                    
                    improvement = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0
                    
                    metric_analysis[f'{treatment_version}_vs_control'] = {
                        'improvement_percent': improvement,
                        'control_mean': control_mean,
                        'treatment_mean': treatment_mean,
                        'significant': abs(improvement) > 5  # Simplified significance test
                    }
            
            analysis['metrics_analysis'][metric_name] = metric_analysis
        
        return analysis
    
    def stop_ab_test(self, test_name: str) -> bool:
        """Stop an A/B test."""
        if test_name in self.active_tests:
            del self.active_tests[test_name]
            logger.info(f"Stopped A/B test: {test_name}")
            return True
        return False

class VersioningTestingSystem:
    """Main versioning and testing system."""
    
    def __init__(self, repo_path: str = "./versions", test_dir: str = "./tests"):
        self.gitops = GitOpsManager(repo_path)
        self.test_runner = TestRunner(test_dir)
        self.ab_test_manager = ABTestManager()
        
        # CI/CD pipeline configuration
        self.pipeline_config = {
            'auto_test_on_commit': True,
            'required_tests': [TestType.UNIT, TestType.INTEGRATION],
            'deployment_stages': [DeploymentStage.TESTING, DeploymentStage.STAGING, DeploymentStage.PRODUCTION],
            'approval_required_for': [DeploymentStage.PRODUCTION]
        }
    
    async def deploy_component(self, component: str, files: Dict[str, str], 
                             description: str, author: str = "AgentAlina",
                             run_tests: bool = True) -> Tuple[Version, List[TestResult]]:
        """Deploy a component with automatic testing."""
        # Create version
        version = self.gitops.create_version(component, files, description, author)
        
        test_results = []
        
        if run_tests:
            # Run unit tests
            unit_result = await self.test_runner.run_unit_tests(component, version.id)
            test_results.append(unit_result)
            
            # Run integration tests if unit tests pass
            if unit_result.status == TestStatus.PASSED:
                integration_result = await self.test_runner.run_integration_tests(
                    [component], [version.id]
                )
                test_results.append(integration_result)
        
        return version, test_results
    
    async def run_full_test_suite(self, component: str, version_id: str) -> Dict[str, TestResult]:
        """Run full test suite for a component."""
        results = {}
        
        # Unit tests
        unit_result = await self.test_runner.run_unit_tests(component, version_id)
        results['unit'] = unit_result
        
        # Integration tests
        integration_result = await self.test_runner.run_integration_tests([component], [version_id])
        results['integration'] = integration_result
        
        # Performance tests
        perf_config = {
            'duration_seconds': 30,
            'concurrent_users': 5,
            'requests_per_second': 50
        }
        perf_result = await self.test_runner.run_performance_tests(component, version_id, perf_config)
        results['performance'] = perf_result
        
        return results
    
    def create_ab_test_for_versions(self, test_name: str, control_version: str, 
                                  treatment_versions: List[str], 
                                  success_metrics: List[str]) -> bool:
        """Create an A/B test comparing different versions."""
        # Equal traffic split by default
        all_versions = [control_version] + treatment_versions
        traffic_per_version = 1.0 / len(all_versions)
        traffic_split = {version: traffic_per_version for version in all_versions}
        
        config = ABTestConfig(
            name=test_name,
            description=f"A/B test comparing {control_version} vs {', '.join(treatment_versions)}",
            control_version=control_version,
            treatment_versions=treatment_versions,
            traffic_split=traffic_split,
            success_metrics=success_metrics,
            duration_hours=24,
            min_sample_size=100
        )
        
        return self.ab_test_manager.create_ab_test(config)
    
    def get_deployment_status(self, component: str) -> Dict[str, Any]:
        """Get deployment status for a component."""
        versions = self.gitops.list_versions(component, limit=10)
        latest_version = versions[0] if versions else None
        
        status = {
            'component': component,
            'latest_version': latest_version.to_dict() if latest_version else None,
            'recent_versions': [v.to_dict() for v in versions[:5]],
            'test_results': []
        }
        
        if latest_version:
            # Get recent test results for latest version
            test_results = self.test_runner.get_test_results(component)
            status['test_results'] = [r.to_dict() for r in test_results[:5]]
        
        return status
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        # Get recent test results
        all_test_results = self.test_runner.get_test_results()
        recent_results = [r for r in all_test_results 
                         if r.started_at > datetime.now() - timedelta(hours=24)]
        
        passed_tests = len([r for r in recent_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in recent_results if r.status == TestStatus.FAILED])
        total_tests = len(recent_results)
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Get active A/B tests
        active_ab_tests = list(self.ab_test_manager.active_tests.keys())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests_24h': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate
            },
            'ab_tests': {
                'active_count': len(active_ab_tests),
                'active_tests': active_ab_tests
            },
            'repository': {
                'total_commits': len(list(self.gitops.repo.iter_commits(max_count=1000))),
                'latest_commit': self.gitops.repo.head.commit.hexsha[:8]
            }
        }

if __name__ == "__main__":
    # Example usage
    async def example_usage():
        # Initialize system
        system = VersioningTestingSystem()
        
        # Deploy a component
        component_files = {
            "prompt_template.yaml": """
name: "example_prompt"
version: "1.0"
template: |
  You are a helpful assistant.
  User query: {query}
  Please provide a helpful response.
parameters:
  temperature: 0.7
  max_tokens: 150
""",
            "config.json": json.dumps({
                "model": "gpt-4",
                "timeout": 30,
                "retry_count": 3
            }, indent=2)
        }
        
        version, test_results = await system.deploy_component(
            "prompt_engine",
            component_files,
            "Initial prompt engine implementation",
            "Developer"
        )
        
        print(f"Deployed version: {version.id}")
        print(f"Test results: {[r.status.value for r in test_results]}")
        
        # Run full test suite
        full_results = await system.run_full_test_suite("prompt_engine", version.id)
        print(f"Full test suite results: {[(k, v.status.value) for k, v in full_results.items()]}")
        
        # Create A/B test
        system.create_ab_test_for_versions(
            "prompt_engine_test",
            version.id,
            [version.id],  # Same version for demo
            ["response_quality", "response_time"]
        )
        
        # Simulate some A/B test data
        for i in range(50):
            user_id = f"user_{i}"
            assigned_version = system.ab_test_manager.assign_user_to_version("prompt_engine_test", user_id)
            
            # Simulate metrics
            system.ab_test_manager.record_metric("prompt_engine_test", user_id, "response_quality", 0.8 + (i % 10) * 0.02)
            system.ab_test_manager.record_metric("prompt_engine_test", user_id, "response_time", 1.0 + (i % 5) * 0.1)
        
        # Analyze A/B test
        ab_analysis = system.ab_test_manager.analyze_ab_test("prompt_engine_test")
        print(f"A/B test analysis: {json.dumps(ab_analysis, indent=2, default=str)}")
        
        # Get system health
        health = system.get_system_health()
        print(f"System health: {json.dumps(health, indent=2)}")
    
    # Run example
    asyncio.run(example_usage())