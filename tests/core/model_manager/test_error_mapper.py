# File: tests/core/model_manager/test_error_mapper.py
from app.core.model_manager.utils.error_mapper import map_model_manager_error
from app.core.model_manager import exceptions as exc


class TestModelManagerErrorMapper:

    def test_map_cuda_out_of_memory(self):
        """RuntimeError with 'CUDA out of memory' -> OutOfVramError"""
        raw_err = RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB...")
        mapped = map_model_manager_error(raw_err)
        assert isinstance(mapped, exc.OutOfVramError)
        assert "Hardware constraint reached" in str(mapped)

    def test_map_standard_memory_error(self):
        """MemoryError -> OutOfVramError"""
        raw_err = MemoryError("Cannot allocate memory")
        mapped = map_model_manager_error(raw_err)
        assert isinstance(mapped, exc.OutOfVramError)

    def test_map_cuda_crash(self):
        """RuntimeError with device-side assert -> CudaCrashError"""
        raw_err = RuntimeError("CUDA error: device-side assert triggered")
        mapped = map_model_manager_error(raw_err)
        assert isinstance(mapped, exc.CudaCrashError)

    def test_map_file_not_found(self):
        """FileNotFoundError -> ModelNotFoundError"""
        raw_err = FileNotFoundError("No such file or directory: 'model.bin'")
        mapped = map_model_manager_error(raw_err)
        assert isinstance(mapped, exc.ModelNotFoundError)

    def test_map_corrupted_weights(self):
        """EOFError -> ModelLoadError"""
        raw_err = EOFError("Ran out of input while deserializing")
        mapped = map_model_manager_error(raw_err)
        assert isinstance(mapped, exc.ModelLoadError)
        assert "corrupted or incompatible" in str(mapped)

    def test_map_fallback_error(self):
        """Generic Exception -> ModelManagerError"""
        raw_err = Exception("Random unexpected bit flip")
        mapped = map_model_manager_error(raw_err)
        assert type(mapped) is exc.ModelManagerError
        assert "unexpected error occurred" in str(mapped)