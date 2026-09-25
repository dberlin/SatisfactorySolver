# Copyright (c) 2024 Daniel Berlin and others
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from importlib import resources
from pathlib import Path
from typing import ClassVar, Self, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    model_validator,
)

from satisfactorysolver.modeler_models import AllDataModel


class _NamedRawRecord(RootModel[dict[str, JsonValue]]):
    @property
    def name(self) -> str:
        name = self.root.get("Name")
        if not isinstance(name, str) or not name:
            raise ValueError("each game data record must have a non-empty string Name")
        return name

    @model_validator(mode="after")
    def validate_name(self) -> Self:
        _ = self.name
        return self


class _RawGameData(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    Machines: list[_NamedRawRecord] = Field(default_factory=list)
    MultiMachines: list[_NamedRawRecord] = Field(default_factory=list)
    Parts: list[_NamedRawRecord] = Field(default_factory=list)
    Recipes: list[_NamedRawRecord] = Field(default_factory=list)


class _MergedGameData(TypedDict):
    Machines: list[dict[str, JsonValue]]
    MultiMachines: list[dict[str, JsonValue]]
    Parts: list[dict[str, JsonValue]]
    Recipes: list[dict[str, JsonValue]]


def _merge_named_records(
    base_records: list[_NamedRawRecord],
    overlay_records: list[_NamedRawRecord],
) -> list[dict[str, JsonValue]]:
    records_by_name = {record.name: record.root.copy() for record in base_records}
    for overlay_record in overlay_records:
        base_record = records_by_name.get(overlay_record.name)
        if base_record is None:
            records_by_name[overlay_record.name] = overlay_record.root.copy()
        else:
            base_record.update(overlay_record.root)
    return list(records_by_name.values())


def _merge_game_data(base: _RawGameData, overlay: _RawGameData) -> _MergedGameData:
    return _MergedGameData(
        Machines=_merge_named_records(base.Machines, overlay.Machines),
        MultiMachines=_merge_named_records(base.MultiMachines, overlay.MultiMachines),
        Parts=_merge_named_records(base.Parts, overlay.Parts),
        Recipes=_merge_named_records(base.Recipes, overlay.Recipes),
    )


def load_game_data(directory: Path | None = None) -> AllDataModel:
    if directory is None:
        data_directory = resources.files("satisfactorysolver").joinpath("data")
    else:
        data_directory = directory

    base = _RawGameData.model_validate_json(
        data_directory.joinpath("game_data.json").read_bytes()
    )
    overlay = _RawGameData.model_validate_json(
        data_directory.joinpath("additional_data.json").read_bytes()
    )
    return AllDataModel.model_validate(_merge_game_data(base, overlay))
