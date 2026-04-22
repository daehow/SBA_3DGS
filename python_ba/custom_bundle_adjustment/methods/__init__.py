"""BA 방법론 모듈 모음.

각 파일은 자기 파일명과 동일한 이름의 함수를 다음 시그니처로 노출해야 한다:

    def {name}(recon: Reconstruction, **kwargs) -> Reconstruction

- `recon`  : custom_bundle_adjustment.bundle_adjustment.Reconstruction
- 반환값  : 최적화된 Reconstruction (같은 객체를 in-place 수정 후 반환해도 됨)
"""
