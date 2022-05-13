import {Autocomplete, Box, Button, Grid, TextField} from "@mui/material";
import {countries, CountryT, FilterFormatT, FilterT, PlatformsFilterT, PlatformsT, PlatformT} from "../../types/types";
import React, {ChangeEvent, SyntheticEvent, useEffect, useState} from "react";
import axios from "axios";


export function Filter(props: { onAnalyze: (data: PlatformsFilterT[]) => void }) {
  const [platforms, setPlatforms] = useState<PlatformsT>({platforms: []});
  const [orFilters, setOrFilters] = useState<PlatformsFilterT[]>([{platform: null, filters: []}]);

  useEffect(() => {
    axios.get('/api/platform').then(response => {
      setPlatforms({
        platforms: Object.entries(response.data as { string: string }).map((p: [string, string], i: number) => {
          return {
            id: p[0],
            name: p[1]
          };
        })
      });
    });
  }, []);

  return (
    <>
      {orFilters.map(orF => {
        return (
          <PlatformFilter platforms={platforms} filters={orF}/>
        );
      })}
      <Button variant="text"
        // color="secondary"
              onClick={() => {
                setOrFilters([...orFilters, {platform: null, filters: []}]);
              }}>or</Button>
      <Button variant="text"
        // color="secondary"
              onClick={() => {
                props.onAnalyze(orFilters);
              }}>Analyze</Button>
    </>
  );
}


function PlatformFilter(props: { platforms: PlatformsT, filters: PlatformsFilterT }) {
  const [selectedPlatform, setSelectedPlatform] = useState<PlatformT>();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedPlatform) {
      setLoading(true)
      axios.get('api/filter?platform=' + selectedPlatform?.id)
        .then(res => {
          props.filters.platform = selectedPlatform.id;
          props.filters.filters = res.data.map((f: any) => {
            return {
              filter_type: f.filter_type,
              // @ts-ignore
              value_format: FilterFormatT[f.value_format]
            }
          });
          setLoading(false)
        });
    }
  }, [selectedPlatform]);

  return (
    <Grid container spacing={2} justifyContent="flex-start"
          alignItems="baseline" width="0.8" margin="auto" border={1}
          borderColor="primary.main"
          borderRadius={2}>
      <Grid item xs>
        <Autocomplete
          disablePortal
          options={props.platforms.platforms}
          getOptionLabel={(p: PlatformT) => p.name}
          onChange={(e: SyntheticEvent, value: PlatformT | null) => value != null ? setSelectedPlatform(value) : null}
          sx={{width: 200}}
          renderInput={(params) => <TextField {...params} label="Platform" variant="standard"
            // color="secondary"
          />}
        />
      </Grid>
      {props.filters.filters.map((f: FilterT) => {
        return (
          <Grid item xs key={f.name}>
            <FilterItem filter={f}/>
          </Grid>
        )
      })}
    </Grid>
  );
}

function FilterItem(props: { filter: FilterT }) {
  const [selectedValue, setSelectedValue] = useState<string | null>(null);
  const [selectedBooleanValue, setSelectedBooleanValue] = useState<boolean>(false);

  useEffect(() => {
    if (props.filter.value_format == FilterFormatT.BOOLEAN) {
      setSelectedValue(selectedBooleanValue ? 'true' : 'false');
    }
    props.filter.value = selectedValue;
  }, [selectedValue, selectedBooleanValue]);

  return (
    <div>
      {
        {
          [FilterFormatT.TEXT]: <TextField
            sx={{width: 200}}
            label={props.filter.name}
            variant="standard"
            // color="secondary"
            onChange={(e: ChangeEvent<HTMLInputElement>) => setSelectedValue(e.target.value)}/>,
          [FilterFormatT.DATE]: <TextField
            sx={{width: 200}}
            label={props.filter.name}
            variant="standard"
            type="date"
            // color="secondary"
            defaultValue={new Date().toLocaleDateString().split('/').reverse().join('-')}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setSelectedValue(e.target.value)}/>,
          [FilterFormatT.COUNTRY]: <Autocomplete
            sx={{width: 200}}
            options={countries}
            autoHighlight
            getOptionLabel={(option) => option.label}
            onChange={(e: SyntheticEvent, value: CountryT | null) => value != null ? setSelectedValue(value.code) : null}
            renderOption={(props, option) => (
              <Box component="li" sx={{'& > img': {mr: 2, flexShrink: 0}}} {...props}>
                <img
                  loading="lazy"
                  width="20"
                  src={`https://flagcdn.com/w20/${option.code.toLowerCase()}.png`}
                  srcSet={`https://flagcdn.com/w40/${option.code.toLowerCase()}.png 2x`}
                  alt=""
                />
                {option.label} ({option.code})
              </Box>
            )}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Choose a country"
                variant="standard"
                // color="secondary"
                inputProps={{
                  ...params.inputProps,
                  autoComplete: 'new-password',
                }}
              />
            )}
          />,
          [FilterFormatT.BOOLEAN]: <Button
            sx={{width: 200}}
            variant={selectedBooleanValue ? 'contained' : 'text'}
            // color="secondary"
            onClick={() => setSelectedBooleanValue(!selectedBooleanValue)}>
            {props.filter.name}
          </Button>
        }[props.filter.value_format]
      }
    </div>
  );
}

