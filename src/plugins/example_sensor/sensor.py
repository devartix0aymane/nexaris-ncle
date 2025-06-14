#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Sensor Implementation for Plugin System
"""

import time
import random
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class ExampleSensor(QObject):
    data_updated = pyqtSignal(dict) # Emits sensor data
    status_changed = pyqtSignal(str)  # Emits status updates

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}
        self.sensor_id = self.config.get("sensor_id", "example_01")
        self.update_interval_ms = self.config.get("update_interval_ms", 1000)
        self.is_active = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_data)

        self.status_changed.emit(f"Sensor {self.sensor_id} initialized.")

    def start(self):
        if not self.is_active:
            self.is_active = True
            self._timer.start(self.update_interval_ms)
            self.status_changed.emit(f"Sensor {self.sensor_id} started.")
            print(f"[ExampleSensor - {self.sensor_id}] Started data acquisition.")
        else:
            self.status_changed.emit(f"Sensor {self.sensor_id} already running.")

    def stop(self):
        if self.is_active:
            self.is_active = False
            self._timer.stop()
            self.status_changed.emit(f"Sensor {self.sensor_id} stopped.")
            print(f"[ExampleSensor - {self.sensor_id}] Stopped data acquisition.")

    def _read_data(self):
        if not self.is_active:
            return

        # Simulate reading data from a sensor
        simulated_data = {
            "timestamp": time.time(),
            "sensor_id": self.sensor_id,
            "type": "simulated_metric",
            "value": random.uniform(20.0, 30.0), # e.g., temperature
            "unit": "Celsius",
            "status": "nominal"
        }
        self.data_updated.emit(simulated_data)
        # print(f"[ExampleSensor - {self.sensor_id}] Emitted data: {simulated_data['value']:.2f}")

    def get_status(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "is_active": self.is_active,
            "update_interval_ms": self.update_interval_ms
        }

    def configure(self, new_config: dict):
        self.config.update(new_config)
        self.sensor_id = self.config.get("sensor_id", self.sensor_id)
        self.update_interval_ms = self.config.get("update_interval_ms", self.update_interval_ms)
        if self.is_active: # Re-apply timer interval if active
            self._timer.setInterval(self.update_interval_ms)
        self.status_changed.emit(f"Sensor {self.sensor_id} reconfigured.")
        print(f"[ExampleSensor - {self.sensor_id}] Reconfigured.")

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    sensor_config = {"sensor_id": "test_sensor_007", "update_interval_ms": 500}
    my_sensor = ExampleSensor(config=sensor_config)

    my_sensor.data_updated.connect(lambda data: print(f"Received data: {data}"))
    my_sensor.status_changed.connect(lambda status: print(f"Status update: {status}"))

    my_sensor.start()

    # Run for a few seconds then stop
    QTimer.singleShot(5000, my_sensor.stop)
    QTimer.singleShot(6000, app.quit)

    sys.exit(app.exec_())