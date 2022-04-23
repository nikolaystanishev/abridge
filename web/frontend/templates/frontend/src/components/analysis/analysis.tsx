import React, {useEffect} from "react";
import {AnalysisT, PlatformsFilterT} from "../../types/types";
import axios from "axios";

export function Analysis(props: { filter: PlatformsFilterT[], reload: boolean }) {
  const [analysis, setAnalysis] = React.useState<AnalysisT>();

  useEffect(() => {
    if (props.reload) {
      axios.post('api/analysis', props.filter).then(response => {
        setAnalysis(response.data);
      });
    }
  }, [props.reload]);

  return (
    <>
    </>
  )
}