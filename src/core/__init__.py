#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core components for NEXARIS Cognitive Load Estimator

This package contains the core functionality for the NEXARIS
Cognitive Load Estimator, including data management, task simulation,
behavior tracking, facial analysis, and cognitive load calculation.
"""

__all__ = [
    'data_manager',
    'task_simulator',
    'behavior_tracker',
    'facial_analyzer',
    'cognitive_load_calculator',
    'eeg_integration'
]

# Import core components
from . import data_manager
from . import task_simulator
from . import behavior_tracker
from . import facial_analyzer
from . import cognitive_load_calculator
from . import eeg_integration

# Import classes for direct access
from .data_manager import DataManager
from .task_simulator import TaskSimulator
from .behavior_tracker import BehaviorTracker
from .facial_analyzer import FacialAnalyzer
from .cognitive_load_calculator import CognitiveLoadCalculator
from .eeg_integration import EEGIntegration