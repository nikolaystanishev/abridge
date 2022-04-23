import {Filter} from "../filter/filter";
import React, {useState} from "react";
import {PlatformsFilterT} from "../../types/types";
import {Analysis} from "../analysis/analysis";

export function Report() {

  const [filter, setFilter] = useState<PlatformsFilterT[]>([]);

  const setAnalyzeData = (data: PlatformsFilterT[]) => {
    setFilter(data);
  };

  return (
    <>
      <Filter onAnalyze={setAnalyzeData}/>
      <Analysis filter={filter}/>
    </>
  )
}