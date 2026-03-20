from app.features.scanner.utils.error_mapper import map_scanner_error
from app.features.scanner import exceptions as exc

class TestScannerErrorMapper:

    def test_map_permission_error(self):
        """PermissionError -> ScanPermissionError"""
        raw_err = PermissionError("Access Denied")
        mapped = map_scanner_error(raw_err, context="/root")

        assert isinstance(mapped, exc.ScanPermissionError)
        assert "/root" in str(mapped)

    def test_map_file_not_found_error(self):
        """FileNotFoundError -> ScanPathNotFound"""
        raw_err = FileNotFoundError("Gone")
        mapped = map_scanner_error(raw_err, context="/missing")

        assert isinstance(mapped, exc.ScanPathNotFound)

    def test_map_generic_os_error(self):
        """OSError -> ScannerError"""
        raw_err = OSError("Disk Full")
        mapped = map_scanner_error(raw_err, context="/disk")

        assert isinstance(mapped, exc.ScannerError)
        # Ensure it's not one of the specific subclasses if we didn't want it to be
        assert not isinstance(mapped, exc.ScanPathNotFound)

    def test_map_unknown_error(self):
        """Exception -> ScannerError"""
        raw_err = Exception("Random Crash")
        mapped = map_scanner_error(raw_err)

        assert isinstance(mapped, exc.ScannerError)
        assert "Unexpected scanner error" in str(mapped)

