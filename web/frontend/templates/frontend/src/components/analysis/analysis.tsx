import React, {useEffect} from "react";
import {AnalysisT, PlatformsFilterT} from "../../types/types";
import axios from "axios";

export function Analysis(props: { filter: PlatformsFilterT[] }) {
  const [analysis, setAnalysis] = React.useState<AnalysisT>();

  useEffect(() => {
    if (props.filter.filter(f => f.filters.length > 0).length > 0) {
      axios.post('api/analysis', props.filter).then(response => {
        setAnalysis(response.data);
      });
    }
  }, [props.filter]);

  return (
    <>
    </>
  )
}